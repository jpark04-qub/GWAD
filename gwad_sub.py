import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from skimage import feature


class Screener:
    def __init__(self, enable=False, thold=0.3, sigma=3, n=10, dim=40):
        self.enable = enable
        self.thold = thold
        self.sigma = sigma # edge detail
        self.n = n # fifo depth
        self.fifo = []
        self.dimension = dim # square dimension
        assert dim % 8 == 0, "dimension must be multiple of 8"
        self.lut = []
        for i in range(8):
            byte = 1 << 7-i
            self.lut.append(byte)

    def reset(self):
        self.fifo.clear()
        return

    def gen_edge(self, x):
        image = x.cpu()
        image = image.squeeze()
        image = image.numpy()

        # image = ndi.gaussian_filter(image, 1)
        edges = feature.canny(image, sigma=self.sigma)
        edges = torch.tensor(edges)
        return edges.flatten()

    def down_size(self, e):
        xe = torch.zeros(int(self.dimension*self.dimension/8))
        idx = 0
        byte = 0
        for i in range(len(e)):
            if i > 0 and i % 8 == 0:
                xe[idx] = byte
                idx += 1
                byte = 0
            if e[i]:
                byte = byte | self.lut[i % 8]
        return xe

    def input_transform(self, x):
        transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((self.dimension, self.dimension))  # edge
        ])
        x = transform(x.clone().detach())

        xe = self.gen_edge(x)
        #xe = self.down_size(e)
        return xe

    def screen(self, e0, e1):
        # bitwise and to get difference
        d = np.sum(e1 ^ e0)
        # rate of difference
        s = np.sum(e0) + np.sum(e1)
        v = d/(s + 0.00000000001)

        # threshold
        flag = False
        if v < self.thold:
            flag = True
        return flag

    def fifo_full(self):
        if len(self.fifo) == self.n:
            return True
        return False

    def fifo_in(self, x):
        if self.fifo_full():
            # fifo is full, shift and empty oldest slot for new example
            self.fifo.pop(0)
        # extract feature
        self.fifo.append(x)
        return

    def run(self, x):
        # pre-process to screen out dummy injections
        suspicious = True

        if not self.enable:
            # pre-process is disabled, always return true to pass example to gwad
            return True

        # transform example
        xn = self.input_transform(x)
        if len(self.fifo) == 0:
            # first query is always screened out
            suspicious = False
        else:
            ei = torch.stack(self.fifo, 0)
            d = torch.sum(ei ^ xn, 1)
            s = torch.sum(ei, 1) + torch.sum(xn)
            dist = d / (s + 0.00000000001)
            score = torch.min(dist)
            if score >= self.thold:
                # no suspicious query exists in the window, so dummy to be screened out
                suspicious = False
        self.fifo_in(xn)
        return suspicious


class ExampleSave:
    def __init__(self, mu, std, format):
        # target model training dataset setting
        self.data_mu = mu
        self.data_std = std
        self.data_foramt = format

    def ready_img(self, x):
        shape = x.size()
        img = x.clone().detach()
        img = img.reshape(shape[1], shape[2], shape[3])
        for i in range(3):
            img[i] = img[i] * self.data_std[i] + self.data_mu[i]

        img = img.cpu()
        if self.data_foramt == 'tensor':
            img *= 255
        else:
            img = img.permute(1, 2, 0).numpy()
            img *= 255
            img = np.uint8(img)
        return img

    def imwrite(self, x, directory, id):
        img = self.ready_img(x, type='numpy')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # assuming the model is pretrained with RGB images.
        cv2.imwrite(os.path.join(directory, '{}.jpg'.format(id)), img)
        return

    def sequence(self, x, id):
        directory = 'tmp/sequence/'
        self.imwrite(x, directory, id)
        return

    def attack_screened(self, x, id):
        directory = 'tmp/attack_screened/'
        self.imwrite(x, directory, id)
        return

    def attack_passed(self, x, id):
        directory = 'tmp/attack_passed/'
        self.imwrite(x, directory, id)
        return

    def dummy_screened(self, x, id):
        directory = 'tmp/dummy_screened/'
        self.imwrite(x, directory, id)
        return

    def dummy_passed(self, x, id):
        directory = 'tmp/dummy_passed/'
        self.imwrite(x, directory, id)
        return