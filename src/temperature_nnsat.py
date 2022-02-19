import os
import csv
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT, TemperatureModel
from data_maker import generate
from config import load_default_config
import mk_problem
from Utils.log_utils import print_log
from confusion import ConfusionMatrix
from problem_loader import ProblemLoader


def load_model(args, log_file=None):
    net = TemperatureModel(NeuroSAT(args))
    net = net.cuda()

    if args.restore:
        if log_file is not None:
            print_log(f'restoring from{args.restore}', f=log_file)
        model = torch.load(args.restore)
        net.load_state_dict(model['state_dict'])

    return net


def predict(net, data):
    net.eval()
    outputs = net(data)
    probs = net.neurosat.vote
    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
    return preds.cpu().detach().numpy(), probs.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    load_default_config(parser)
    parser.add_argument('--output-dir', type=str, default=None, help='val file dir')
    parser.add_argument('--temperature', '-t', type=float, default=None, help='temperature')
    args = parser.parse_args()
    log_file = open(os.path.join(args.log_dir, args.task_name + '.log'), 'a+')

    acc_output_file_path = f'{args.output_dir}/{args.data_dir.split("/")[-1]}.txt'

    net = load_model(args)

    net.temperature = nn.Parameter(torch.Tensor(np.array([args.temperature])).cuda())
    print(net.temperature)

    conf_mat = ConfusionMatrix()
    times = []

    test_problems_loader = ProblemLoader(args.data_dir)
    acc_list = []
    while test_problems_loader.has_next():
        xs = test_problems_loader.get_next()
        times = []
        for x in tqdm(xs):
            start_time = time.time()
            preds, probs = predict(net, x)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            times.append(duration)

            # target = np.array(x.is_sat)
            preds = torch.Tensor(preds)
            target = torch.Tensor(x.is_sat)
            conf_mat.update(preds, target)
        acc_list.append(conf_mat.get_acc())
        desc = "%s - %d rnds: tot time %.2f ms for %d cases, avg time: %.2f ms; %s" \
               % (f'{args.data_dir.split("/")[-1]}_{args.n_rounds}', args.n_rounds, sum(times), len(times),
                  sum(times) * 1.0 / len(times), conf_mat)
        print_log(desc)

    print(f'{args.data_dir.split("/")[-1]}_{args.n_rounds}_acc: {np.mean(np.array(acc_list))}')
    with open(acc_output_file_path, 'a+') as f:
        f.write(f'{conf_mat.get_acc()} ')
