import torch
import numpy as np
import torch.nn.functional as F
from ....base import HardLabelAttackBase

"""
Chen et. al. 
Boosting Decision based Black-box Adversarial Attacks with Random Sign Flip
European Conference on Computer Vision (ECCV), 2020
"""

class SignFlip(HardLabelAttackBase):
    def __init__(self, device, model, lp='l2', q_budgets=[1000], eps=0.01, stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='dec')
        self.name = 'sign-flip'

        self.min = 0
        self.max = 1
        self.epsilon = eps

        self.converged = False
        self.converg_eps = eps

        self.alpha = 0.004 * 0.1
        self.init_try = 1000 # iteration to find initial theta

        self.x = 0
        self.prob = 0
        self.prob_ub = 0.9999
        self.prob_lb = 0.999

        self.scale = 1
        self.reset = 0
        self.proj_success = 0
        self.flip_success = 0

        self.zo_cnt = 0
        self.ls_cnt = 0
        self.no_cnt = 0

    def resize(self, x, h, w):
        return F.interpolate(x, size=[h, w], mode='bilinear', align_corners=False)

    def distance(self, x_adv):
        diff = x_adv.reshape(x_adv.size(0), -1)
        if self.lp == 'inf':
            out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        else:
            out = torch.sqrt(torch.sum(diff * diff)).item()
        return out

    def project_infinity(self, x_a, x, l):
        return torch.max(x - l, torch.min(x_a, x + l))

    def binary_infinity(self, x_adv, k):
        lb = torch.zeros(1).cuda()
        ub = torch.max(torch.abs(x_adv - self.x))
        for _ in range(k):
            mid = (lb + ub) / 2
            adv = self.project_infinity(x_adv, self.x, mid)
            adv = torch.clamp(adv, self.min, self.max)
            self.ls_cnt += 1
            check = self.check_adversary(adv)
            if check:
                ub = mid
            else:
                lb = mid
        x_adv = self.project_infinity(x_adv, self.x, ub)
        x_adv = torch.clamp(x_adv, self.min, self.max)
        return x_adv

    def init_sample(self, x):
        #x_adv = x.clone()
        flag = False
        for _ in range(self.init_try):
            x_adv = torch.rand_like(x).to(self.device)
            x_adv = torch.clamp(x_adv, self.min, self.max)
            self.no_cnt += 1
            if self.check_adversary(x_adv):
                flag = True
                break

        if self.debug:
            if flag:
                print("Found initial example")
            else:
                print("Couldn't find valid initial, failed")

        x_adv = self.binary_infinity(x_adv, 10)
        return flag, x_adv

    def update_param(self):
        self.proj_success = self.proj_success/self.reset
        self.flip_success = self.flip_success/self.reset
        if self.proj_success > 0.7:
            self.alpha = self.alpha * 1.5
        if self.proj_success < 0.3:
            self.alpha = self.alpha / 1.5

        if self.alpha > 1e+20:
            self.alpha = 1e+20
        if self.alpha < 1e-20:
            self.alpha = 1e-20

        if self.flip_success > 0.7:
            self.prob = self.prob - 0.001
        if self.flip_success < 0.3:
            self.prob = self.prob + 0.001
        self.prob = torch.clamp(self.prob, self.prob_lb, self.prob_ub)
        self.reset = 0
        self.proj_success = 0
        self.flip_success = 0
        return

    def perturb(self, x_best):
        self.reset += 1
        delta = x_best - self.x
        h_dr = self.shape[2]//self.scale
        w_dr = self.shape[3]//self.scale

        eta = torch.randn([self.shape[0], self.shape[1], h_dr, w_dr])
        eta = eta.sign() * self.alpha
        eta = self.resize(eta, self.shape[2], self.shape[3])
        eta = eta.to(self.device)

        l = torch.max(delta)

        ex0 = delta + eta
        ex1 = torch.zeros_like(eta)
        delta_p = self.project_infinity(ex0, ex1, l-self.alpha)

        x_adv = self.x + delta_p
        x_adv = torch.clamp(x_adv, self.min, self.max)
        self.zo_cnt += 1
        if self.check_adversary(x_adv):
            delta = delta_p
            self.proj_success += 1

        s = torch.bernoulli(self.prob) * 2 - 1
        delta_s = delta * self.resize(s, self.shape[2], self.shape[3])
        x_adv = self.x + delta_s
        x_adv = torch.clamp(x_adv, self.min, self.max)
        self.zo_cnt += 1
        if self.check_adversary(x_adv):
            self.prob = (-1) * s * 1e-4
            self.prob = torch.clamp(self.prob, self.prob_lb, self.prob_ub)
            self.flip_success += 1
            delta = delta_s

        if self.reset%10 == 0:
            self.update_param()

        #l = torch.max(delta)

        #if l < self.converg_eps:
        #    self.converged = True

        x_best = self.x + delta

        return x_best

    def attack(self, sm, x_best):
        if sm == 0:
            flag, x_best = self.init_sample(x_best)
            if flag:
                sm = 1
            self.query_cnt2 = self.query_cnt
        elif sm == 1:
            x_best = self.perturb(x_best)

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone()
        self.x = x_
        x_best = self.x

        self.prob = torch.ones_like(x_)
        self.prob = self.prob * 0.999
        self.prob = self.resize(self.prob, self.shape[2]//self.scale, self.shape[3]//self.scale)

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

