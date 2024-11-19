import torch
import numpy as np
import matplotlib.pyplot as plt


class Utility:
    def __init__(self, num=1):
        return
        self.num = num
        self.item = []
        for i in range(num):
            self.list = []
            self.item.append(self.list)

    def get(self, prob):
        return
        for i in range(self.num):
            self.item[i].append(prob[0, i])

    def show(self):

        return
        plt.figure(figsize=(5, 5))
        for i in range(self.num):
            plt.plot(self.item[i], label=i)

        plt.legend()
        plt.show()



