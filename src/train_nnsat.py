import os
import pickle
import argparse
import mk_problem
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary

from neurosat import NeuroSAT, TemperatureModel
from data_maker import generate
from config import load_default_config
from Utils.log_utils import print_log
from confusion import ConfusionMatrix
from problem_loader import ProblemLoader


def T_scaling(logits, temperature):
    return sigmoid(torch.div(logits, temperature))


def _eval():
    global desc, loss
    loss = loss_fn(T_scaling(outputs, temperature), target)

    # desc = 'loss: %.4f; ' % (loss.item())
    loss.backward(retain_graph=True)
    # retain_graph = True
    return loss


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


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    load_default_config(parser)
    args = parser.parse_args()

    task_name = f'{args.task_name}_sr{args.min_n}to{args.max_n}_pairs{args.n_pairs}_ep{args.epochs}_nr{args.n_rounds}_d{args.dim}'

    log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')
    detail_log_file = open(os.path.join(args.log_dir, task_name + '_detail.log'), 'a+')

    # loading data
    print(f'loading data...')
    print(f'Data Dir is: {args.data_dir}')
    print(f'Train data dir is: {args.train_file}')
    train_problems_loader = ProblemLoader(args.train_file)
    print(f'Val data dir is: {args.val_file}')
    val_problems_loader = ProblemLoader(args.val_file)
    print(f'loading data done\n')

    # init NNSAT
    print(f'init NNSAT...')
    tNet = TemperatureModel(NeuroSAT(args))
    tNet = tNet.cuda()

    loss_fn = nn.BCELoss()
    opt = MultipleOptimizer(optim.Adam([{'params': tNet.neurosat_params}], lr=0.00002, weight_decay=1e-10),
                            optim.Adam([{'params': tNet.temperature}], lr=0.0001, weight_decay=1e-10))
                            # optim.LBFGS([{'params': tNet.temperature}], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe'))
    print(f'init NNSAT done\n')
    sigmoid = nn.Sigmoid()
    best_acc = 0.0
    start_epoch = 0

    # for tmp in range(500):
    #     print()
    #     print()
    #     tNet.train()
    #     for p in range(1):
    #         target = torch.Tensor(train[p].is_sat).cuda().float()
    #         print(tNet.temperature)
    #         opt.optimizers[1].zero_grad()
    #         outputs = tNet(train[p])
    #         loss = loss_fn(sigmoid(outputs), target)
    #         print(loss.item())
    #         loss.backward()
    #         # def closure():
    #         #     opt.optimizers[1].zero_grad()
    #         #     outputs = tNet(train[p])
    #         #     loss = loss_fn(sigmoid(outputs), target)
    #         #     print(loss.item())
    #         #     loss.backward()
    #         #     return loss
    #         # opt.optimizers[1].step(closure)
    #         opt.optimizers[1].step()
    #         print(tNet.temperature)
    #
    #         opt.optimizers[0].zero_grad()
    #         outputs = tNet(train[p])
    #         loss = loss_fn(sigmoid(outputs), target)
    #         loss.backward()
    #         opt.optimizers[0].step()

    ##################################
    # # torchsummary.summary(net)
    train_loss = []
    val_loss_l = []
    temp_list = []
    for epoch in range(start_epoch, args.epochs):
        print_log('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc),
                  f=[log_file, detail_log_file])
        train = train_problems_loader.get_next()
        train_bar = tqdm(train)
        train_mat = ConfusionMatrix()
        tNet.train()

        lossInE = []
        for _, prob in enumerate(train_bar):
            target = torch.Tensor(prob.is_sat).cuda().float()
            opt.optimizers[1].zero_grad()
            outputs = tNet(prob)
            loss = loss_fn(sigmoid(outputs), target)
            loss.backward()
            opt.optimizers[1].step()


            # def closure():
            #     opt.optimizers[1].zero_grad()
            #     outputs = tNet(prob)
            #     loss = loss_fn(outputs, target)
            #     loss.backward()
            #     return loss
            # opt.optimizers[1].step(closure)

            opt.optimizers[0].zero_grad()
            outputs = tNet(prob)
            loss = loss_fn(sigmoid(outputs), target)
            loss.backward()
            lossInE.append(loss.item())
            opt.optimizers[0].step()

            preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

            train_mat.update(preds, target)

        epLoss = np.mean(np.array(lossInE))
        train_loss.append(epLoss)
        print(f'loss in epoch {epoch + 1}: {epLoss} | {train_mat}')
        print(f'epoch {epoch} T: {tNet.temperature.item()}')
        temp_list.append(tNet.temperature.item())

        # val stage
        print(f'\nepoch {epoch} val:')
        val = val_problems_loader.get_next()
        val_bar = tqdm(val)
        val_mat = ConfusionMatrix()
        tNet.eval()
        valLossInE = []
        for _, prob in enumerate(val_bar):
            outputs = tNet(prob)
            target = torch.Tensor(prob.is_sat).cuda().float()
            val_loss = loss_fn(sigmoid(outputs), target)
            valLossInE.append(val_loss.item())
            preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

            val_mat.update(preds, target)

        epLoss = np.mean(np.array(valLossInE))
        print(f'val loss in epoch {epoch + 1}: {epLoss} | {val_mat}')
        val_loss_l.append(epLoss)

        acc = val_mat.get_acc()
        torch.save({'epoch': epoch+1, 'acc': acc, 'state_dict': tNet.state_dict(), 'temperature': tNet.temperature}, os.path.join(args.model_dir, task_name+'_last.pth.tar'))
        if acc >= best_acc:
            best_acc = acc
            torch.save({'epoch': epoch+1, 'acc': best_acc, 'state_dict': tNet.state_dict(), 'temperature': tNet.temperature}, os.path.join(args.model_dir, task_name+'_best.pth.tar'))

    train_loss = np.array(train_loss)
    val_loss_l = np.array(val_loss_l)
    temp_list = np.array(temp_list)
    with open(os.path.join(args.model_dir, f'{task_name}_loss_npy'), 'wb') as f:
        np.save(f, train_loss)
        np.save(f, val_loss_l)
    with open(os.path.join(args.model_dir, f'{task_name}_temp.npy'), 'wb') as f:
        np.save(f, temp_list)
