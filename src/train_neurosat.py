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

from neurosat import NeuroSAT
from data_maker import generate
from config import load_default_config
from Utils.log_utils import print_log
from confusion import ConfusionMatrix
from problem_loader import ProblemLoader


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
    print(f'Train data dir is: {args.train_file}')
    train_problems_loader = ProblemLoader(args.train_file)
    print(f'Val data dir is: {args.val_file}')
    val_problems_loader = ProblemLoader(args.val_file)
    print(f'loading data done\n')

    # init NNSAT
    print(f'init NNSAT...')

    net = NeuroSAT(args)
    net = net.cuda()

    loss_fn = nn.BCELoss()
    print(f'init NNSAT done\n')

    sigmoid = nn.Sigmoid()
    best_acc = 0.0
    start_epoch = 0

    nOpt = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
    # torchsummary.summary(net)
    train_loss = []
    val_loss_l = []

    for epoch in range(start_epoch, args.epochs):
        print_log('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, args.epochs, best_acc),
                  f=[log_file, detail_log_file])
        train = train_problems_loader.get_next()
        train_bar = tqdm(train)
        train_mat = ConfusionMatrix()
        net.train()
        lossInE = []
        for _, prob in enumerate(train_bar):
            nOpt.zero_grad()
            outputs = net(prob)
            target = torch.Tensor(prob.is_sat).cuda().float()

            loss = loss_fn(sigmoid(outputs), target)
            loss.backward()
            lossInE.append(loss.item())
            nOpt.step()

            preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
            train_mat.update(preds, target)

        epLoss = np.mean(np.array(lossInE))
        print(f'train loss in epoch {epoch + 1}: {epLoss} | {train_mat}')
        train_loss.append(epLoss)

        # val stage
        val = val_problems_loader.get_next()
        val_bar = tqdm(val)
        val_mat = ConfusionMatrix()
        net.eval()
        valLossInE = []
        for _, prob in enumerate(val_bar):
            outputs = net(prob)
            target = torch.Tensor(prob.is_sat).cuda().float()
            val_loss = loss_fn(sigmoid(outputs), target)
            valLossInE.append(val_loss.item())
            preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
            val_mat.update(preds, target)

        epLoss = np.mean(np.array(valLossInE))
        print(f'val loss in epoch {epoch + 1}: {epLoss} | {val_mat}')
        val_loss_l.append(epLoss)

        acc = val_mat.get_acc()
        torch.save({'epoch': epoch+1, 'acc': acc, 'state_dict': net.state_dict()}, os.path.join(args.model_dir, task_name+'_last.pth.tar'))
        if acc >= best_acc:
            best_acc = acc
            torch.save({'epoch': epoch+1, 'acc': best_acc, 'state_dict': net.state_dict()}, os.path.join(args.model_dir, task_name+'_best.pth.tar'))

    train_loss = np.array(train_loss)
    val_loss_l = np.array(val_loss_l)
    with open(os.path.join(args.model_dir, f'{task_name}_loss_npy'), 'wb') as f:
        np.save(f, train_loss)
        np.save(f, val_loss_l)
