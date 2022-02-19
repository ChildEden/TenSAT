import numpy as np

import torch


class ConfusionMatrix(object):

    def __init__(self):
        self.TP = torch.zeros(1).long()
        self.TN = torch.zeros(1).long()
        self.FN = torch.zeros(1).long()
        self.FP = torch.zeros(1).long()
        self.total = torch.zeros(1).long()

    def update(self, preds, target):
        # print(self.TP.item(), self.TN.item(), self.FN.item(), self.FP.item(), self.total.item())
        self.TP += (preds.eq(1) & target.eq(1)).cpu().sum()
        self.TN += (preds.eq(0) & target.eq(0)).cpu().sum()
        self.FN += (preds.eq(0) & target.eq(1)).cpu().sum()
        self.FP += (preds.eq(1) & target.eq(0)).cpu().sum()
        self.total = self.TP + self.TN + self.FN + self.FP

    def get_acc(self):
        return (self.TP.item() + self.TN.item()) * 1.0 / self.total.item()

    def __str__(self):
        return f'TP:{self.TP.item()* 1.0/self.total.item()}, TN:{self.TN.item()* 1.0/self.total.item()}, FN:{self.FN.item()* 1.0/self.total.item()}, FP:{self.FP.item()* 1.0/self.total.item()}, acc:{self.get_acc()}'

    def __repr__(self):
        return self.__str__()
