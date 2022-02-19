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

from neurosat import NeuroSAT
from data_maker import generate
from config import load_default_config
import mk_problem
from Utils.log_utils import print_log
from confusion import ConfusionMatrix
from problem_loader import ProblemLoader
from sklearn.metrics import f1_score


def load_model(args, log_file=None):
    net = NeuroSAT(args)
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
    probs = net.vote
    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())
    return preds.cpu().detach().numpy(), probs.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    load_default_config(parser)
    parser.add_argument('--output-dir', type=str, default=None, help='val file dir')
    args = parser.parse_args()

    acc_output_file_path = f'{args.output_dir}/{args.data_dir.split("/")[-1]}.txt'

    net = load_model(args)
    test_problems_loader = ProblemLoader(args.data_dir)
    solved_count = 0
    sat_count = 0
    while test_problems_loader.has_next():
        xs = test_problems_loader.get_next()
        times = []
        for x in tqdm(xs):
            start_time = time.time()
            # net.find_solutions(x)
            solves = net.find_solutions(x)
            solvesP = np.array(list(map(lambda i: i is not None, solves)))
            sat_list = np.array(x.is_sat)
            solved = solvesP[np.where(sat_list == True)]
            solved_count += np.sum(solved)
            sat_count += len(solved)

            end_time = time.time()
            duration = (end_time - start_time) * 1000
            times.append(duration)

    acc = solved_count / sat_count
    print(
        "%s - %d rnds: tot time %.2f ms for %d cases, avg time: %.2f ms; acc: %s" \
        % (f'{args.data_dir.split("/")[-1]}_{args.n_rounds}', args.n_rounds, sum(times), len(times),
           sum(times) * 1.0 / len(times), str(acc))
    )
    with open(acc_output_file_path, 'a+') as f:
        f.write(f'{acc} ')
