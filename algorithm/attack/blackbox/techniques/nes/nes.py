import torch
import numpy as np
from ....base import SoftLabelAttackBase
import sys
import time

"""
Ilyas et. al. 
Black-box adversarial attacks with limited queries and information
International Conference on Machine Learning, 2018
"""

class NES(SoftLabelAttackBase):
    def __init__(self, device, model, lp='l2', q_budgets=[1000], eps=255, lr=2.55, sigma=0.1, stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='inc')
        self.name = 'nes'

        self.epsilon = eps
        self.sigma = sigma #0.1 #2.55
        self.lr = lr #1.5 #2.55 #2.55 #2.55 # learning rate
        self.min = 0
        self.max = 1

        self.nes_iter = 50
        self.x = []   # original input

        self.zo_cnt = 0
        self.ls_cnt = 0
        self.no_cnt = 0

    def criterion(self, logit, y):
        # cw loss
        if self.targeted:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(y)
            second_max_index = target_is_max.long() * argsort[:, 1] + (~ target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), y]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
            _, argsort = logit.sort(dim=1, descending=True)
            gt_is_max = argsort[:, 0].eq(y)
            second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), y]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def predict_loss(self, x):
        x = x.clone()

        bound = torch.max(torch.abs(self.min), torch.abs(self.max))
        x = torch.clamp(x-self.x, bound*(-1), bound) + self.x
        x = torch.clamp(x, self.min, self.max)

        sigma = 0
        #x = x + sigma * torch.randn_like(x)
        #x = torch.clamp(x, self.min, self.max)

        logit, _ = self.probability(x, self.label)
        loss = self.criterion(logit, self.label)
        return loss.detach()

    def lp_step(self, x, g):
        x = x.clone()
        if self.lp == 'inf':
            x = x + self.lr * g.sign()
        else:
            norm_g = torch.norm(g)
            if norm_g != 0:
                x = x + self.lr * g / torch.norm(g)
        return x.detach()

    def perturb(self, x_adv):
        x_adv = x_adv.clone()
        g = torch.zeros_like(x_adv)
        for _ in range(self.nes_iter):
            ui = torch.randn_like(x_adv)
            ui = self.varying_random_vector(ui)
            px = x_adv + self.sigma * ui
            nx = x_adv - self.sigma * ui
            self.zo_cnt += 2
            g += self.predict_loss(px) * ui
            g -= self.predict_loss(nx) * ui
        g = g/(2*self.nes_iter*self.sigma)
        # perform the step
        x_adv = self.lp_step(x_adv, g)
        return x_adv.detach()

    def project(self, x_adv):
        if self.lp == 'inf':
            x_adv = torch.clamp(x_adv, self.x-self.epsilon, self.x+self.epsilon)
        else:
            delta = x_adv - self.x
            norm_d = torch.norm(delta)
            v0 = (norm_d <= self.epsilon).float() * delta
            if norm_d != 0:
                v1 = (norm_d > self.epsilon).float()*self.epsilon*delta/norm_d
            else:
                v1 = torch.zeros_like(delta)
            x_adv = self.x + v0 + v1
        return torch.clamp(x_adv, self.min, self.max)

    def check(self, x, y):
        x = x.clone()
        x = torch.clamp(x, self.min, self.max)
        p = self.prediction(x)
        return p.item() != y.item()

    def attack(self, sm, x_best):
        x_adv = x_best.clone()

        x_adv_cap = self.perturb(x_adv)
        x_adv_cap = self.project(x_adv_cap)

        x_best = x_adv_cap

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone()
        self.x = x_
        x_best = self.x

        return x_best

    def core(self, image):
        x_best = self.setup(image)

        # Generate the adversarial samples
        x_adv, queries0, queries1 = self.run(x_best)

        return x_adv, queries0, queries1

    def untarget(self, image, label):
        self.targeted = False
        self.label = label
        self.min = torch.min(image.flatten())
        self.max = torch.max(image.flatten())

        img = image.detach().clone()
        adv, q0, q1 = self.core(img)

        return adv, q0, q1, self.adaptive.query_cnt

    def target(self, image, label, example):
        raise ValueError("targeted attack is not supported yet")
        return None

