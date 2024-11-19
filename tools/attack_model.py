import torch
import numpy as np
from tools.attack_methods import Method
import torchvision.transforms as T


class AttackModel():
    def __init__(self, data_type, device, stats, cfg, d_model=None, q_budgets=[], stop=True):
        super().__init__()
        self.name = 'delta_test'
        self.device = device
        self.stats = stats
        self.method = Method(data_type, device, cfg, d_model=d_model, q_budgets=q_budgets, stop=stop)

    def update_stats(self, x, adv, y, predict):
        _, self.stats.hx = predict.data.max(1)
        self.stats.dist0 = torch.norm(adv - x)
        self.stats.dist1 = (torch.norm(adv - x)) / (torch.norm(x) + 1e-17)
        self.stats.dist0_sum += self.stats.dist0
        self.stats.dist1_sum += self.stats.dist1
        if self.stats.hx != y:
            bin_size = (0.4 / self.stats.num_bins)
            d_idx = torch.floor(self.stats.dist1 / bin_size)
            d_idx = int(d_idx.cpu().numpy())
            if d_idx > self.stats.num_bins:
                d_idx = self.stats.num_bins
            self.stats.dist1_hist[d_idx] += 1
            self.stats.successful += 1
            if self.stats.dist1 < 0.1:
                self.stats.pass_num += 1
        return

    def set_adaptive(self, adapt_type, adapt_rate, x2_pool):
        self.method.alg.set_adaptive(adapt_type, adapt_rate, x2_pool)

    def run(self, x, y):
        adv, self.stats.iter0, self.stats.iter1, self.stats.iter2 = self.method.run(x, y)
        return adv