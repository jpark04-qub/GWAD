import torch
import numpy as np
from ....base import HardLabelAttackBase

"""
J. Chen, M. I. Jordan, and M. J. Wainwright. 
Hopskipjumpattack: A query-efficient decision-based attack. 
IEEE Symposium on Security and Privacy, 2020
"""


class HSJA(HardLabelAttackBase):
    def __init__(self, device, model, lp='l2', q_budgets=[1000], stop=True):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='dec')
        self.name = 'hsja'

        self.min = 0
        self.max = 1

        self.theta = 0.01
        self.init_eval = 100
        self.threshold = 0.001
        self.max_eval = np.max(q_budgets)
        self.iter_cnt = 0

        self.x = 0

        self.zo_cnt = 0
        self.ls_cnt = 0
        self.no_cnt = 0

    def _interpolate(self, current_sample, original_sample, alpha):

        if self.lp == 'l2':
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            lb = torch.min((original_sample - alpha).flatten())
            hb = torch.max((original_sample + alpha).flatten())
            result = torch.clamp(current_sample, lb, hb)

        return result

    def _compute_delta(self, current_sample, original_sample):
        # Note: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        if self.iter_cnt == 0:
            return 0.1 * (self.max - self.min)

        if self.lp == 'l2':
            dist = torch.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(current_sample.size())) * self.theta * dist
        else:
            dist = torch.max(torch.abs(original_sample - current_sample))
            delta = np.prod(current_sample.size()) * self.theta * dist

        return delta

    def _binary_search(self, current_sample, original_sample, threshold):
        # First set upper and lower bounds as well as the threshold for the binary search
        if self.lp == 'l2':
            upper_bound = torch.tensor(1)
            lower_bound = torch.tensor(0)

            if threshold is None:
                threshold = self.theta

        else:
            upper_bound = torch.max(torch.abs(original_sample - current_sample))
            upper_bound = upper_bound.cpu()
            lower_bound = torch.tensor(0)

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample, original_sample, alpha)

            # Update upper_bound and lower_bound
            self.ls_cnt += 1
            satisfied = self.check_adversary(interpolated_sample)

            lower_bound = torch.where(torch.tensor(satisfied == 0), alpha, lower_bound)
            upper_bound = torch.where(torch.tensor(satisfied == 1), alpha, upper_bound)

        result = self._interpolate(current_sample, original_sample, upper_bound)

        return result

    def _init_sample(self):
        x = self.x.clone()
        initial_sample = None

        if self.targeted:
            print("targeted attack is not supported yet")
            return None
        else:
            for _ in range(self.init_eval):
                random_img = torch.rand(x.size()).to(self.device)
                random_img = random_img.uniform_(self.min, self.max)
                #random_img = self.varying_random_vector(random_img)
                self.no_cnt += 1
                random_class = self.prediction(random_img.reshape(self.shape))

                if random_class != self.label:
                    # Binary search to reduce the l2 distance to the original image
                    random_img = self._binary_search(random_img, x, self.threshold)
                    initial_sample = random_img
                    break

        return initial_sample

    def _compute_update(self, current_sample, num_eval, delta):
        rnd_noise_shape = [num_eval] + list(current_sample.size())
        if self.lp == 'l2':
            rnd_noise = torch.randn(rnd_noise_shape)
            for i in range(num_eval):
                rnd_noise[i, :] = self.varying_random_vector(rnd_noise[i, :])
        else:
            rnd_noise = torch.rand(rnd_noise_shape)
        rnd_noise = rnd_noise.to(self.device)

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / torch.sqrt(
            torch.sum(rnd_noise ** 2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True))

        eval_samples = torch.clamp(current_sample + delta * rnd_noise, self.min, self.max)
        rnd_noise = (eval_samples.cpu() - current_sample.cpu()) / (delta.cpu()+1e-32)

        # Compute gradient: This is a bit different from the original paper, instead we keep those that are
        # implemented in the original source code of the authors
        satisfied = torch.zeros(num_eval)
        for i in range(num_eval):
            self.zo_cnt += 1
            satisfied[i] = self.check_adversary(eval_samples[i, :].reshape(self.shape))

        del eval_samples

        f_val = 2 * satisfied.cpu() - 1.0
        f_val = f_val

        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rnd_noise, axis=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rnd_noise, axis=0)
        else:
            f_val -= torch.mean(f_val)
            f_val = f_val.reshape([len(f_val), 1])
            grad = torch.mean(f_val * rnd_noise, axis=0)

        # Compute update
        if self.lp == 'l2':
            result = grad / (torch.norm(grad)+1e-32)
        else:
            result = torch.sign(grad)

        return result.to(self.device)

    def perturb(self, x_best):

        # Set current perturbed image to the initial image
        original_sample = self.x.clone()
        current_sample = x_best.clone()

        # First compute delta
        delta = self._compute_delta(current_sample, original_sample)

        # Then run binary search
        current_sample = self._binary_search(current_sample, original_sample, self.threshold)

        # Next compute the number of evaluations and compute the update
        num_eval = min(int(self.init_eval * np.sqrt(self.iter_cnt + 1)), self.max_eval)
        num_eval = min(num_eval, np.abs(self.query_max-self.query_cnt))

        update = self._compute_update(current_sample, num_eval, delta)

        # Finally run step size search by first computing epsilon
        if self.lp == 'l2':
            dist = torch.norm(original_sample - current_sample)
        else:
            dist = torch.max(torch.abs(original_sample - current_sample))

        epsilon = 2.0 * dist / np.sqrt(self.iter_cnt + 1)
        success = False

        temp_cnt = 0
        while not success:
            epsilon /= 2.0
            potential_sample = current_sample + epsilon * update
            self.ls_cnt += 1
            success = self.check_adversary(torch.clamp(potential_sample, self.min, self.max))
            if success:
                break
            temp_cnt += 1
            if temp_cnt >= 200:
                break

        # Update current sample
        if success is True:
            current_sample = torch.clamp(potential_sample, self.min, self.max)

        # Update current iteration
        self.iter_cnt += 1

        # If attack failed. return original sample
        if torch.isnan(current_sample).any():  # pragma: no cover
            x_best = x_best
        else:
            x_best = current_sample

        return x_best

    def attack(self, sm, x_best):
        if sm == 0:
            x_best = self._init_sample()
            if x_best is not None:
                sm = 1
            self.query_cnt2 = self.query_cnt
        elif sm == 1:
            x_best = self.perturb(x_best)

        return sm, x_best

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone().flatten()
        self.x = x_
        x_best = torch.zeros_like(self.x)

        # Set binary search threshold
        if self.lp == 'l2':
            self.theta = 0.01 / np.sqrt(np.prod(self.shape))
        else:
            self.theta = 0.01 / np.prod(self.shape)

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


