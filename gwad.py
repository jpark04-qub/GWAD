import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from gwad_sub import ExampleSave, Screener


def extract_ds(x0, x1, x2):
    # feature normalisation strategy
    # 1. dimensionality normalisation
    #   - (discarded) convert to grayscale
    #   - convert to low dimension space using bilinear resizing method
    transform = T.Compose([
        #T.Grayscale(num_output_channels=1),
        T.Resize((32, 32))
    ])
    x0 = transform(x0).flatten().cpu()
    x1 = transform(x1).flatten().cpu()
    x2 = transform(x2).flatten().cpu()

    # 2. distribution normalisation
    #   - transpose to normal distribution with mu = 0.0
    #   - (discarded) standardise the length of vectors. unit vector
    x0 = x0 - torch.mean(x0)
    x1 = x1 - torch.mean(x1)
    x2 = x2 - torch.mean(x2)

    # calculate delta feature
    d0 = (x1 - x0)
    d1 = (x2 - x1)

    # delta similarity - cosine angle of two delta
    dot_product = torch.matmul(d0, d1)
    d0_norm = torch.norm(d0)
    d1_norm = torch.norm(d1)
    if d0_norm == 0 or d1_norm == 0:
        raw_angle = torch.tensor(1e-32)
    else:
        raw_angle = dot_product / (d0_norm * d1_norm)
    abs_angle = torch.abs(raw_angle)

    if torch.isnan(abs_angle):
        print("warning : angle is nan!!")

    ds = raw_angle
    return ds, abs_angle, d0_norm, d1_norm


class GWAD():
    def __init__(self, device, cfg, stats, mode='defence', model=None, delta_net=None):
        super().__init__()
        self.name = 'gwad_v5'
        self.device = device
        self.stats = stats
        self.mode = mode
        self.model = model
        self.delta_net = delta_net
        self.example_save = ExampleSave(cfg["model"]["data_mu"], cfg["model"]["data_std"], cfg["model"]["data_format"])
        enable = cfg["screen"]["on"]
        thold = cfg["screen"]["thold"]
        fifo_depth = cfg["screen"]["n"]
        self.screener = Screener(enable=enable, thold=thold, sigma=3, n=fifo_depth, dim=32)

        # example queue to calculate ds
        self.example_q = []

        # ds configurations
        self.ds_size = 16*16
        self.ds_q = []
        self.ds_out = 0
        self.ds_in = 0

        # hods configuration
        self.hods_bins = 200
        self.hods_vector = [0]
        self.hods_flag = False

        self.show_detect = True
        self.query_cnt = 0
        self.delta_query_cnt = 0

        self.classes = delta_net.classes
        self.hx = torch.zeros(7)

        self.stats.classes = self.classes
        self.query_to_check = []

        self.suspicious = False

    def defended_model(self, x):
        self.query_cnt += 1
        return self.model(x)

    def delta_model(self, feat):
        self.delta_query_cnt += 1
        self.delta_net.eval()
        output = self.delta_net(feat)
        _, hx = output.data.max(1)
        return hx

    def reset(self):
        self.stats.reset()
        self.query_cnt = 0
        self.delta_query_cnt = 0
        self.hx = torch.zeros(7)
        self.show_detect = True
        self.screener.reset()
        self.example_q.clear()
        self.ds_q.clear()
        self.hods_flag = False
        print("\nGWAD_%s starts" % (self.mode))
        return

    def examples_ready(self):
        # check if three examples are queued
        if len(self.example_q) == 3:
            return True
        return False

    def examples_queue(self, x):
        if self.examples_ready():
            # if examples queue is full shift and empty oldest slot for new example
            self.example_q.pop(0)

        self.suspicious = True
        # check if example is suspicious
        self.suspicious = self.screener.run(self.example_save.ready_img(x))
        if self.suspicious:
            # suspicious query, pass to gwad
            self.example_q.append(x)
        return

    def dss_ready(self):
        length = len(self.ds_q)
        if length == self.ds_size:
            return True
        elif length > self.ds_size:
            return True
        else:
            return False

    def ds_queue(self, ds):
        # if ds queue is full shift and empty oldest slot for new ds
        if self.dss_ready():
            self.ds_in = ds
            self.ds_out = self.ds_q.pop(0)
        self.ds_q.append(ds)
        return

    def set_hods_element(self, ele, action):
        bin = (-1) - ele
        bin = torch.abs(bin)
        bin_size = 2 / self.hods_bins
        b_idx = bin // bin_size
        if action == 'add':
            self.hods_vector[0, int(b_idx)] += 1
        elif action == 'remove':
            if self.hods_vector[0, int(b_idx)] > 0:
                self.hods_vector[0, int(b_idx)] -= 1
        else:
            raise ValueError('Invalid hods setting action')

    def make_hods(self):
        if self.dss_ready():
            feature = self.ds_q
            # generate historgram of delta similarity
            if self.hods_flag is False:
                # create hods feature vector
                self.hods_flag = True
                self.hods_vector = torch.zeros(1, self.hods_bins + 1)
                for i in range(len(feature)):
                    self.set_hods_element(feature[i], 'add')
                self.hods_vector = self.hods_vector.to(self.device)
            else:
                # hods feature vector has been created, just modify hods with old and new ds.
                self.set_hods_element(self.ds_out, 'remove')
                self.set_hods_element(self.ds_in, 'add')
        return

    def attack_detect(self):
        flag = False
        if self.dss_ready():
            # normalise hods
            m0 = torch.min(self.hods_vector)
            m1 = torch.max(self.hods_vector)
            hods = (self.hods_vector - m0) / (m1 - m0)

            # Delta-Net predicts the type of attack
            hx = self.delta_model(hods)
            hx = hx.cpu()
            del hods

            # check Delta-Net prediction
            if self.classes[hx] == 'benign':
                # failed - benign
                flag = False
            else:
                flag = True
                if self.show_detect:
                    print("GWAD_%s : first detection [%s] made @ %dth query" % (self.mode, self.classes[hx], self.query_cnt))
                    self.show_detect = False

            if self.mode == 'defence' or self.mode == 'simulate':
                self.hx[hx] += 1
        return flag

    def calculate_ds(self):
        # get three consecutive examples from the queue
        x0 = self.example_q[0]
        x1 = self.example_q[1]
        x2 = self.example_q[2]

        # get delta similarity
        ds, _, d0_norm, _ = extract_ds(x0, x1, x2)

        return ds, d0_norm

    def save_query(self, q):
        #self.example_save.sequence(q.x, self.query_cnt)
        if q.t == 'attack':
            # attack query is injected
            if self.suspicious:
                # correctly detected attack so pass to follows
                self.stats.attack_passed += 1
                #self.example_save.attack_passed(q.x, self.query_cnt)
            else:
                # error - attack is seen to be dummy
                self.stats.attack_screened += 1
                #self.example_save.attack_screened(q.x, self.query_cnt)
        elif q.t == 'benign':
            # dummy query is injected
            if self.suspicious:
                # error - dummy is seen to be attack
                self.stats.dummy_passed += 1
                #self.example_save.dummy_passed(q.x, self.query_cnt)
            else:
                # correctly detected dummy so screen out
                self.stats.dummy_screened += 1
                #self.example_save.dummy_screened(q.x, self.query_cnt)
        else:
            UserWarning("aa")
        return

    def run(self, q):
        # q.t - query type: 'dummy' or 'attack'
        # q.x - example

        # get example
        x = q.x.clone().detach()

        if self.mode != 'simulate':
            # query to target model to defend
            output = self.defended_model(x)
        else:
            output = []

        # see enabled
        if self.mode == 'off':
            return output

        # -- attack detection routine --
        # pre-process and example queue
        self.examples_queue(x)

        # save query and update pre-process stats
        if self.mode == 'defence':
            self.save_query(q)

        # check example queue is full for detection
        if self.examples_ready() is False:
            return output

        # calculate delta similarity
        ds, norm = self.calculate_ds()

        # update ds vector
        self.ds_queue(ds)

        # make and update hods feature vector
        self.make_hods()

        # update stats
        self.stats.update(ds, norm)

        # check under attack
        self.attack_detect()
        return output

    def get_stats(self, image):
        x = image.clone().detach()
        update = self.check_update(x)
        return update

    def get_predictions(self):
        return self.hx

    def get_queries(self):
        return self.query_cnt, self.delta_query_cnt

    def show_predictions(self):
        print("GWAD_%s : prediction result - delta-net predicted total {} ".format(self.mode, torch.sum(self.hx)))
        for i in range(len(self.classes)):
            print("%s[%d], " % (self.classes[i], self.hx[i]), end='')
        print("")
        return
