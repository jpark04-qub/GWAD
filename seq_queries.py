import os
import torch
import time
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools.statistics import GWADStatistics, AttackStatistics
from tools.attack_model import AttackModel as ATTACK_MODEL
from gwad import GWAD
from algorithm.attack.base import Query


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    img = img.cpu()
    return img.detach().numpy()


class Statistics:
    def __init__(self):
        self.gwad = GWADStatistics()
        self.attack = AttackStatistics()
        return

    def increment_runs(self):
        self.gwad.increment_runs()
        self.attack.increment_runs()
        return


def show_predicrions(gwad_def, hx_record, total_queries):
    # show gwad prediction record
    hx = gwad_def.get_predictions()
    hx = hx.numpy()
    queries, delta_queries = gwad_def.get_queries()
    total_queries += delta_queries
    print("GWAD_defence : delta-net predicted {} HoDS during {} queries".format(int(np.sum(hx)), queries))
    print("GWAD_defence : accumulated predictions {}/{}".format(int(np.sum(hx_record) + np.sum(hx)), total_queries))
    for i in range(len(hx_record)):
        hx_record[i] += hx[i]
        print("%s[%d], " % (gwad_def.classes[i], hx_record[i]), end='')
    print("")

    return total_queries


def save_distributions(stats_dir, model, stats, feat_type, data_type):
    hist = stats.gwad.mean_hist()
    hist_file = '%s/hist/%s_%s_%s' % (stats_dir, data_type, model.name, feat_type)
    np.savetxt('%s.txt' % (hist_file), hist, delimiter=',')

    dist = stats.gwad.distribution
    dist_file = '%s/dist/%s_%s_%s' % (stats_dir, data_type, feat_type, stats.gwad.runs)
    np.savetxt('%s.txt' % (dist_file), dist, delimiter=',')


def benign(device, cfg, data_type, model, delta_net, loader):
    stats_dir = 'stats'
    cfg_gwad = cfg["gwad"]
    attack_name = 'benign'

    # ready statistics
    stats = Statistics()

    # instantiate gwad
    gwad_def = GWAD(device, cfg_gwad, stats.gwad, mode='defence', model=model, delta_net=delta_net)

    # reset gwad
    gwad_def.reset()

    # main loop
    hx_record = np.zeros(len(delta_net.classes))
    total_queries = 0
    idx = 0
    for data, true_class in loader:
        if idx%10 == 0:
            print(end='\r')
            print("{}/{}".format(idx, len(loader)), end='')
        data = data.to(device)
        q = Query(t='benign', x=data)
        gwad_def.run(q)
        idx += 1

    stats.increment_runs()

    # show gwad prediction record
    show_predicrions(gwad_def, hx_record, total_queries)

    # show defence statistics
    stats.gwad.show_screen()

    # get and save mean histogram in text file
    save_distributions(stats_dir, model, stats, attack_name, data_type)


def attack(device, cfg, data_type, model, delta_net, loader):
    stats_dir = 'stats'
    cfg_attack, cfg_gwad = cfg["attack"], cfg["gwad"]
    attack_name = cfg_attack["name"]
    adapt_type = cfg_attack["adaptive"]["name"]
    batch_size = cfg_attack["adaptive"]["batch_size"]
    pool_size = cfg_attack["adaptive"]["pool_size"]
    move_rate = cfg_attack["adaptive"]["move_rate"]
    q_budgets = [cfg_attack["query_budget"]]

    # ready statistics
    stats = Statistics()

    # instantiate gwad models
    # - defence model  : to generate adversarial examples
    # - evaluate model : to inject generated adversarial example
    gwad_def = GWAD(device, cfg_gwad, stats.gwad, mode='defence', model=model, delta_net=delta_net)
    gwad_evl = GWAD(device, cfg_gwad, stats.gwad, mode='evaluate', model=model, delta_net=delta_net)

    # show setting
    print("\nsequence of queries : attack \n{} {} {} {}".format(data_type, model.name, attack_name, delta_net.name))

    # main loop
    hx_record = np.zeros(len(delta_net.classes))
    total_gwad_predict = 0
    total_attack_queries = 0
    x2_pool = []
    cnt = 0
    for data, true_class in loader:
        data, true_class = data.to(device), true_class.to(device)

        # make benign example pool for multi-agent adaptive attack
        if adapt_type == 'batch':
            if len(x2_pool) < pool_size:
                x2_pool.append(data)
                continue

        # reset gwad
        gwad_def.reset()

        # instantiate attack model to conduct fresh attack
        attack = ATTACK_MODEL(data_type, device, stats.attack, cfg_attack,
                              d_model=gwad_def.run, q_budgets=q_budgets, stop=False)

        # set adaptive attack
        attack.set_adaptive(adapt_type, move_rate, batch_size, x2_pool)

        # perform adversarial attack on current data
        t0 = time.perf_counter()
        adv = attack.run(data, true_class)
        t1 = time.perf_counter()

        # predict adversarial example
        q = Query(t='attack', x=adv[0])
        predict = gwad_evl.run(q)

        # update attack statistics
        attack.update_stats(data, adv[0], true_class, predict)

        # increment count
        stats.increment_runs()

        # show gwad prediction record
        total_gwad_predict = show_predicrions(gwad_def, hx_record, total_gwad_predict)

        # show attack statistics
        print("Attack stats : num - time [true] : [adv][val/suc, dist, ratio, [i1] [i2] [i3]]")
        print("{} - {:.3f}s [{}] : ".format(stats.gwad.runs, t1-t0, true_class.item()), end='')
        stats.attack.show_stats()

        cnt += 1
        total_attack_queries += stats.attack.iter0[0]
        print("average attack queries - {}".format(int(total_attack_queries/cnt)))

        # show defence statistics
        stats.gwad.show_screen()

        # get and save mean histogram in text file
        save_distributions(stats_dir, model, stats, attack_name, data_type)