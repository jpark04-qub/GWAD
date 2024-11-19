import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    img = img.cpu()
    return img.detach().numpy()


class AttackStatistics:
    def __init__(self, number_bins=40):
        self.runs = 0
        self.num_bins = number_bins
        self.successful = 0
        self.pass_num = 0
        self.hx = 0
        self.dist0 = 0
        self.dist1 = 0
        self.dist0_sum = 0
        self.dist1_sum = 0
        self.dist1_hist = np.zeros(number_bins+1)
        self.iter0 = 0
        self.iter1 = 0
        self.iter0_sum = 0
        self.iter1_sum = 0
        self.iter2 = 0
        self.iter2_sum = 0
        return

    def increment_runs(self):
        self.runs += 1
        return

    def show_stats(self):
        print("[{}][{}/{}, {:.3f} {:.3f}, {} {} [{}]]".format(self.hx.item(),
                                                      int(self.pass_num),
                                                      int(self.successful),
                                                      self.dist0, self.dist1,
                                                      self.iter0, self.iter1, self.iter2))

    def show_hist(self):
        print("[", end='')
        for i in range(len(self.dist1_hist)):
            print('{}, '.format(int(self.hist[i])), end='')
        print("]")
        return


class GWADStatistics:
    def __init__(self, num_bins=1000):
        self.runs = 0
        self.cnt = 0

        # Detailed histogram of delta similarity over attack (not HoDS)
        self.num_bins = num_bins*2
        self.hist = torch.zeros(num_bins*2+1) # accumulated histogram, average = histogram/runs

        # DS distribution over attack
        self.distribution = []

        # pre-process screening performance
        self.dummy_screened = 0
        self.attack_screened = 0
        self.dummy_passed = 0
        self.attack_passed = 0

        self.successful = 0
        self.classes = []
        self.predictions = []
        return

    def reset(self):
        self.cnt = 0
        self.distribution.clear()
        return

    def increment_runs(self):
        self.runs += 1
        return

    def update(self, ds, norm):
        # update overall histogram. note it is not HoDS
        bin = (-1) - ds
        bin = torch.abs(bin)
        bin_size = 2 / self.num_bins
        b_idx = bin // bin_size

        self.hist[int(b_idx)] += 1

        # update ds distribution
        self.distribution.append(ds)
        self.cnt += 1
        return

    def mean_hist(self):
        hist = []
        if self.runs == 0:
            print("number of runs must be non-zero")
        else:
            for i in range(len(self.hist)):
                hist.append(self.hist[i] / self.runs)
        return hist

    def show_stats(self):
        return

    def show_screen(self):
        print("GWAD screen : [screened dummy] [screened attack] [passed dummy] [passed attack]")
        print("[{}] [{}] [{}] [{}]".format(self.dummy_screened, self.attack_screened, self.dummy_passed, self.attack_passed))

    def show_ds_hist(self):
        print("hist")
        for i in range(len(self.hist)):
            print('{}, '.format(self.hist[i] / self.runs), end='')
        print("")
        return

    def show_ds_distribution(self):
        print("list")
        for i in range(len(self.distribution)):
            print('{}, '.format(self.distribution[i]), end='')
        print("")
        return
