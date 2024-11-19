import torch
import numpy as np


class Base:
    def __init__(self, device='cpu'):
        self.device = device

    def forward(self):
        print('Attack Base forward')


class WhiteBoxAttackBase:
    def __init__(self, device='cpu'):
        self.device = device

    def forward(self):
        print('Attack Base forward')


class Query:
    def __init__(self, t='benign', x=None):
        self.t = t
        self.x = x


class AdaptiveAttacks:
    def __init__(self, device='cpu'):
        self.device = device
        self.x2_action = 0
        self.x2_pool0 = []
        self.x2_pool1 = []
        self.x2_seq = []
        self.x2_size = 0
        self.x2_idx = 0
        self.x2_factor = 4.5
        self.query_cnt = 0

    def reset(self, pool):
        # pool is split into  two sub-pools to avoid duplicate examples
        # in the last window and the first window after re-shuffled
        size = int(np.round(len(pool)/2))
        self.x2_pool0 = pool[0:size]
        self.x2_pool1 = pool[size:size*2]
        self.x2_seq = torch.randperm(size)
        self.x2_idx = 0
        self.x2_action = 0 # sub-pool id to use
        self.x2_size = size

    def varying_random_mean(self, eta, rate):
        m0 = torch.min(eta)
        m1 = torch.max(eta)

        # positive bound of mean - ratio of distribution variance
        mean = (m1-m0)*rate

        # sign of mean
        sign = torch.randn(1).to(eta.device)
        sign /= torch.abs(sign)

        # move mean of distribution
        eta = eta + sign*mean
        return eta

    def varying_random_variance(self, eta, rate):
        m0 = torch.min(eta)
        m1 = torch.max(eta)

        m3 = torch.var(eta)

        sigma = torch.rand(1).to(eta.device)
        if sigma > 1.0:
            sigma = 1.0
        # positive bound of mean - ratio of distribution variance
        sigma *= rate
        # sign of mean
        sign = torch.randn(1).to(eta.device)
        sign /= torch.abs(sign)

        r = sign*sigma
        if r > rate or r < (-1)*rate:
            sigma = sigma

        sigma = 1+sign*sigma
        if sigma < 0:
            sigma = 1e-10

        eta = eta * sigma

        m4 = torch.var(eta)
        return eta

    def multi_agent(self, model, x, n):
        print("multi agent mode is not supported!")
        return model(x, 'attack')

    def dummy_benign(self, model, x, n):
        # [r_i benign examples] -> [x] -> [r_j benign examples]
        #r = torch.rand(1)
        #r = torch.round(r * n)
        r = n
        for i in range(int(r)):
            if self.x2_idx >= self.x2_size:
                self.x2_idx = 0
                self.x2_seq = torch.randperm(self.x2_size)
                self.x2_action ^= 1
            x2_idx = self.x2_seq[self.x2_idx]
            # dummy query with benign example affects defence only
            # need to use two pools to avoid example duplication may happen after re-shuffle
            if self.x2_action == 0:
                x2 = self.x2_pool0[x2_idx]
            else:
                x2 = self.x2_pool1[x2_idx]

            _ = model(x2.to(self.device), 'benign')

            self.x2_idx += 1
            self.query_cnt += 1
        # true query for attack
        output = model(x, 'attack')

        return output


# common attack base function
class BlackBoxAtackCommonBase:
    def __init__(self, device='cpu', model=None, lp='l2', stop=True, q_budgets=[10000], perturbation='dec'):
        self.decice = device
        self.model = model
        self.lp = lp
        self.stop = stop
        self.q_budgets = np.sort(q_budgets)
        self.query_max = np.max(q_budgets)
        self.extension = True # attack finishes current iteration after budget is over
        self.perturbation = perturbation # perturbation direction: NES = increase, HSJA = decrease
        self.adaptive = AdaptiveAttacks(device)
        self.adapt_type = None
        self.adapt_rate = 0.1
        self.query_data = Query()

    def model_(self, x, type):
        self.query_cnt += 1
        self.query_data.t = type
        self.query_data.x = x
        return self.model(self.query_data)

    def inject(self, x):
        if self.adapt_type == 'batch':
            # dummy input with benign.
            output = self.adaptive.dummy_benign(self.model_, x, self.adapt_rate)
        else:
            output = self.model_(x, 'attack')
        return output

    def prediction(self, x):
        output = self.inject(x)
        _, hx = output.data.max(1)
        return hx

    def probability(self, x, y):
        output = self.inject(x)
        prob = output[0, y]
        return output, prob

    def check_adversary(self, x):
        x = x.clone()
        hx = self.prediction(x.reshape(self.shape))
        flag = False
        if self.targeted:
            if hx == self.label:
                flag = True
        else:
            if hx != self.label:
                flag = True
        return flag

    def set_adaptive(self, adapt_type, adapt_rate, x2_pool):
        self.adapt_type = adapt_type
        self.adapt_rate = adapt_rate
        self.adaptive.reset(x2_pool)

    def varying_random_vector(self, eta):
        if self.adapt_type == 'rm':
            # variation in mean of random distribution
            eta = self.adaptive.varying_random_mean(eta, self.adapt_rate)
        elif self.adapt_type == 'rv':
            # variation in variance of random distribution
            eta = self.adaptive.varying_random_variance(eta, self.adapt_rate)
        return eta

    def stop_criteria(self, x, x_adv):
        if x_adv is None:
            return False
        x = x.clone().detach().reshape(self.shape)
        x_adv = x_adv.clone().detach().reshape(self.shape)
        flag = False
        eta = x_adv - x
        eta_norm = torch.norm(eta)
        rho = eta_norm / (torch.norm(x)+1e-17)
        if self.stop:
            if self.perturbation == 'inc':
                # perturbation increases as iteration grows
                if self.check_adversary(x_adv):
                    flag = True
            else:
                # perturbation decreases as iteration grows
                if self.check_adversary(x_adv) and (rho <= self.max_rho):
                    flag = True
        return flag

    def run(self, x_best):
        x_adv = []
        queries0 = []
        queries1 = []
        sm = 0
        for i in range(len(self.q_budgets)):
            self.query_max = self.q_budgets[i]
            self.extension = True
            while True:
                sm, x_best = self.attack(sm, x_best)
                if self.stop_criteria(self.x, x_best):
                    break
                if self.query_cnt >= self.query_max:
                    break
            self.max_rho *= 0.9
            if x_best is None:
                x_best = self.x
            x_best = torch.clamp(x_best, self.min, self.max)
            x_adv.append(x_best.reshape(self.shape))
            queries0.append(self.query_cnt)
            queries1.append(self.query_cnt2)

        return x_adv, queries0, queries1


class SoftLabelAttackBase(BlackBoxAtackCommonBase):
    def __init__(self, device='cpu', targeted=False, model=None, lp='l2', stop=True, q_budgets=[10000], max_rho=0.1,
                 debug=False, perturbation='inc'):
        super().__init__(device=device,
                         model=model,
                         lp=lp,
                         stop=stop,
                         q_budgets=np.sort(q_budgets),
                         perturbation=perturbation)
        self.device = device
        self.targeted = targeted
        self.query_cnt = 0
        self.query_cnt2 = 0
        self.shape = torch.zeros(4) # [batch_size x channel x width x height]
        self.label = 0
        self.max_rho = max_rho
        self.debug = debug

    def attack(self, sm, x_best):
        raise NotImplementedError


class HardLabelAttackBase(BlackBoxAtackCommonBase):
    def __init__(self, device='cpu', targeted=False, model=None, lp='l2', stop=True, q_budgets=[10000], max_rho=0.1,
                 debug=False, perturbation='dec'):
        super().__init__(device=device,
                         model=model,
                         lp=lp,
                         stop=stop,
                         q_budgets=np.sort(q_budgets),
                         perturbation=perturbation)
        self.device = device
        self.targeted = targeted
        self.query_cnt = 0
        self.query_cnt2 = 0
        self.shape = torch.zeros(4) # [batch_size x channel x width x height]
        self.label = 0
        self.stop = stop
        self.max_rho = max_rho
        self.debug = debug


    def attack(self, sm, x_best):
        raise NotImplementedError

