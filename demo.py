import torch
import seq_queries
from utilities.load_data import load_data, ConfigDataLoad
from utilities.load_model import get_models
from tools.delta_nets import get_delta_net
import json
from argparse import ArgumentParser

"""
This Demo version of GWAD only supports CIFAR10, but not restricted. models and weights are downloaded from 
https://github.com/huyvnphan/PyTorch_CIFAR10
Modify load_data and load_model in utilities to add other models and dataset     
"""
data_type = 'cifar10'

""" check cuda is available """
print(torch.__version__)
print(f"Is CUDA supported by this system? {torch.cuda.is_available()} ")
print(f"CUDA version: {torch.version.cuda}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_cfg(cfg):
    cfg_attack, cfg_gwad = cfg["attack"], cfg["gwad"]
    if cfg_gwad["screen"]["on"] and cfg_attack["adaptive"]["name"] == 'batch':
        batch_size = cfg_attack["adaptive"]["batch_size"]
        fifo_depth = cfg_gwad["screen"]["n"]
        assert batch_size < fifo_depth, "screener fifo depth must be greater than batch size"
        pool_size = cfg_attack["adaptive"]["pool_size"]
        assert fifo_depth*2 <= pool_size, "benign pool size must be greater than fifo depth x 2 to avoid duplications"


def main(args):
    json_file = "configs/" + args.data + "/" + args.attack + "/" + "cfg_" + args.scenario + ".json"
    cfg = json.load(open(json_file))
    check_cfg(cfg)

    """ load threat model """
    target_model = get_models(data_type, cfg["gwad"]["model"]["name"])
    target_model.eval()
    target_model.to(device)

    """ load datasat """
    data_load_cfg = ConfigDataLoad(args.data)
    data_load_cfg.train.shuffle = True
    data_load_cfg.train.batch_size = 1
    data_load_cfg.test.shuffle = False
    data_load_cfg.test.batch_size = 1
    train_ldr, test_ldr, classes, _ = load_data(data_load_cfg)

    """ load gwad classifier """
    delta_net = get_delta_net(device)

    """ test scenario """
    if args.scenario == "benign":
        seq_queries.benign(device, cfg, args.data, target_model, delta_net, test_ldr)
    else:
        seq_queries.attack(device, cfg, args.data, target_model, delta_net, test_ldr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--attack', type=str)
    parser.add_argument('--scenario', type=str)

    arg = parser.parse_args()

    main(parser.parse_args())


