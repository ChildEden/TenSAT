import pandas as pd
import argparse
import pickle
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT
from data_maker import generate
import mk_problem

from config import load_default_config

parser = argparse.ArgumentParser()
load_default_config(parser)
args = parser.parse_args()

net = NeuroSAT(args)
net = net.cuda()

task_name = args.task_name + '_sr' + str(args.min_n) + 'to' + str(args.max_n) + '_ep' + str(args.epochs) + '_nr' + str(
    args.n_rounds) + '_d' + str(args.dim)
log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')
detail_log_file = open(os.path.join(args.log_dir, task_name + '_detail.log'), 'a+')


def T_scaling(logits, temperature):
    return torch.div(logits, temperature)


temperature = nn.Parameter(torch.ones(1).cuda())

train, val = None, None

if args.train_file is not None:
    with open(os.path.join(args.data_dir, 'train', args.train_file), 'rb') as f:
        train = pickle.load(f)

with open(os.path.join(args.data_dir, 'val', args.val_file), 'rb') as f:
    print('Data Dir is: %s' % (args.data_dir))
    val = pickle.load(f)


def _eval():
    global desc, loss
    loss = loss_fn(sigmoid(T_scaling(outputs, temperature)), target)
    # desc = 'loss: %.4f; ' % (loss.item())
    loss.backward(retain_graph=True)
    # retain_graph = True
    return loss


torch.autograd.set_detect_anomaly(True)


#### Two Optimiser method #####
class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        opt_adam = self.optimizers[0]
        opt_LBFGS = self.optimizers[1]

        opt_LBFGS.step(_eval)

        opt_adam.step()

    ##################################


###### SCALING ###################
criterion = nn.CrossEntropyLoss()
# optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
##################################
loss_fn = nn.BCELoss()
# pa = list(net.parameters()) + list([temperature])
# optimA = optim.Adam(pa, lr=0.00002, weight_decay=1e-10)
opt = MultipleOptimizer(optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10),
                        optim.LBFGS([temperature], lr=0.1, max_iter=10, line_search_fn='strong_wolfe'))
sigmoid = nn.Sigmoid()
best_acc = 0.0
start_epoch = 0

if train is not None:
    print('num of train batches: ', len(train), file=log_file, flush=True)

print('num of val batches: ', len(val), file=log_file, flush=True)

if args.restore is not None:
    print('restoring from', args.restore, file=log_file, flush=True)
    model = torch.load(args.restore)
    start_epoch = model['epoch']
    best_acc = model['acc']
    net.load_state_dict(model['state_dict'])

##### Temp Scaling Optimizer #####
temps = []
losses = []

##################################

for e in range(1):
    net.train()
    for p in range(1):
        opt.zero_grad()
        outputs = net(train[p])
        # target = torch.Tensor(train[p].is_sat).cuda().float()
        # print()
        # print()
        # print(temperature)
        # opt.step()
        # print(temperature)


# for epoch in range(start_epoch, args.epochs):
#
#     if args.train_file is None:
#         print('generate data online', file=log_file, flush=True)
#         train = generate(args)
#
#     print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc))
#     print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc), file=log_file, flush=True)
#     print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc), file=detail_log_file, flush=True)
#     train_bar = tqdm(train)
#     TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
#     net.train()
#     temps = []
#     losses = []
#
#     for _, prob in enumerate(train_bar):
#         opt.zero_grad()
#         outputs = net(prob)
#         target = torch.Tensor(prob.is_sat).cuda().float()
#         # print(outputs.shape, target.shape)
#         # print(outputs, target)
#         # outputs = sigmoid(outputs)
#         ret = opt.step()
#         # loss.backward()
#         # opt.step()
#
#         preds = torch.where(torch.div(outputs, temperature) > 0.5, torch.ones(outputs.shape).cuda(),
#                             torch.zeros(outputs.shape).cuda())
#
#         TP += (preds.eq(1) & target.eq(1)).cpu().sum()
#         TN += (preds.eq(0) & target.eq(0)).cpu().sum()
#         FN += (preds.eq(0) & target.eq(1)).cpu().sum()
#         FP += (preds.eq(1) & target.eq(0)).cpu().sum()
#         TOT = TP + TN + FN + FP
#
#     # print(desc, file=log_file, flush=True)
#     print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
#     print('Final T_scaling Loss: {:.2f}'.format(loss))
#     val_bar = tqdm(val)
#     TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
#     net.eval()
#     for _, prob in enumerate(val_bar):
#         opt.zero_grad()
#         outputs = net(prob)
#         target = torch.Tensor(prob.is_sat).cuda().float()
#         # print(outputs.shape, target.shape)
#         # print(outputs, target)
#         # outputs = sigmoid(outputs)
#         preds = torch.where(torch.div(outputs, temperature) > 0.5, torch.ones(outputs.shape).cuda(),
#                             torch.zeros(outputs.shape).cuda())
#
#         TP += (preds.eq(1) & target.eq(1)).cpu().sum()
#         TN += (preds.eq(0) & target.eq(0)).cpu().sum()
#         FN += (preds.eq(0) & target.eq(1)).cpu().sum()
#         FP += (preds.eq(1) & target.eq(0)).cpu().sum()
#         TOT = TP + TN + FN + FP
#
#         desc = 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
#         (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),
#         FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
#         # val_bar.set_description(desc)
#         if (_ + 1) % 100 == 0:
#             print(desc, file=detail_log_file, flush=True)
#     print(desc, file=log_file, flush=True)
#
#     acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
#     torch.save({'epoch': epoch + 1, 'acc': acc, 'state_dict': net.state_dict()},
#                os.path.join(args.model_dir, task_name + '_last.pth.tar'))
#     if acc >= best_acc:
#         best_acc = acc
#         torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': net.state_dict()},
#                    os.path.join(args.model_dir, task_name + '_best.pth.tar'))
