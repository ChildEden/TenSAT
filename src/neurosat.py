import os
import csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mlp import MLP


class TemperatureModel(nn.Module):
    def __init__(self, neurosat):
        super(TemperatureModel, self).__init__()
        self.neurosat = neurosat
        self.neurosat_params = self.neurosat.parameters()
        self.temperature = nn.Parameter(torch.ones(1).cuda())
        self.sigmoid = nn.Sigmoid()

    def forward(self, problem):
        vote = self.neurosat(problem)
        return torch.div(vote, self.temperature)


class NeuroSAT(nn.Module):
    def __init__(self, args):
        super(NeuroSAT, self).__init__()
        self.args = args
        self.init_random_seeds()

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = nn.Linear(1, args.dim)

        self.L_msg = MLP(self.args.dim, self.args.dim, self.args.dim)
        self.C_msg = MLP(self.args.dim, self.args.dim, self.args.dim)

        if self.args.rnn_type == 'rnn':
            self.L_update = nn.RNN(self.args.dim * 2, self.args.dim)
            self.C_update = nn.RNN(self.args.dim, self.args.dim)
        if self.args.rnn_type == 'gru':
            self.L_update = nn.GRU(self.args.dim * 2, self.args.dim)
            self.C_update = nn.GRU(self.args.dim, self.args.dim)
        if self.args.rnn_type == 'lstm':
            self.L_update = nn.LSTM(self.args.dim * 2, self.args.dim)
            self.C_update = nn.LSTM(self.args.dim, self.args.dim)
        # self.L_norm   = nn.LayerNorm(self.args.dim)
        # self.C_norm   = nn.LayerNorm(self.args.dim)

        self.L_vote = MLP(self.args.dim, self.args.dim, 1)

        self.denom = torch.sqrt(torch.Tensor([self.args.dim]))

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.is_sat)
        # print(n_vars, n_lits, n_clauses, n_probs)

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        init_ts = self.init_ts.cuda()
        # 1 x n_lits x dim & 1 x n_clauses x dim
        L_init = self.L_init(init_ts).view(1, 1, -1)
        # print(L_init.shape)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        # print(C_init.shape)
        C_init = C_init.repeat(1, n_clauses, 1)

        # print(L_init.shape, C_init.shape)

        L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).to_dense().cuda()

        # normalized adjacency matrix
        # NLocalSAT: Boosting Local Search with Solution Prediction (2020)
        sum0 = torch.diag(torch.reciprocal(torch.sqrt(torch.sum(L_unpack, dim=0)) + 1e-6))
        sum1 = torch.diag(torch.reciprocal(torch.sqrt(torch.sum(L_unpack, dim=1)) + 1e-6))
        normal_L_unpack = torch.mm(torch.mm(sum1, L_unpack), sum0)

        if self.args.is_normal == 1:
            L_unpack = normal_L_unpack

        # print(ts_L_unpack_indices.shape)
        first = 1
        counter = 0
        '''
        l1old = np.zeros(128)
        l1nold  = np.zeros(128)
        l2old  = np.zeros(128)
        l2nold  = np.zeros(128)
        l3old  = np.zeros(128)
        l3nold  = np.zeros(128)
        '''
        for _ in range(self.args.n_rounds):
            # n_lits x dim
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
            # LC_msg = torch.mm(L_unpack.t(), L_pre_msg)
            LC_msg = torch.matmul(L_unpack.t(), L_pre_msg)

            if self.args.rnn_type == 'rnn' or self.args.rnn_type == 'gru':
                _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state[0])
                C_state = (C_state, 0)
            if self.args.rnn_type == 'lstm':
                _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
            # _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)

            # n_clauses x dim
            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
            CL_msg = torch.matmul(L_unpack, C_pre_msg)
            # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)

            if self.args.rnn_type == 'rnn' or self.args.rnn_type == 'gru':
                _, L_state = self.L_update(
                    torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state[0])
                L_state = (L_state, 0)
            if self.args.rnn_type == 'lstm':
                _, L_state = self.L_update(
                    torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)
            # print('L_state', L_state[0].shape, L_state[1].shape)

            ###FOR EXCEL
            logits = L_state[0].squeeze(0)
            clauses = C_state[0].squeeze(0)

        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        # print(logits.shape, clauses.shape)
        vote = self.L_vote(logits)
        self.all_votes = vote  # for solving
        self.final_lits = logits  # for solving
        # print('vote', vote.shape)
        vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
        # print(vote_join)
        # print('vote_join', vote_join.shape)

        self.vote = vote_join
        vote_join = vote_join.view(n_probs, -1, 2).view(n_probs, -1)
        vote_mean = torch.mean(vote_join, dim=1)
        # print('mean', vote_mean.shape)
        return vote_mean

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)

    def init_random_seeds(self):
        torch.manual_seed(self.args.tc_seed)
        torch.cuda.manual_seed(self.args.tc_seed)
        np.random.seed(self.args.np_seed)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def find_solutions(self, problem):
        def flip_vlit(vlit):
            if vlit < problem.n_vars:
                return vlit + problem.n_vars
            else:
                return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        self.forward(problem)

        all_votes = self.all_votes
        final_lits = self.final_lits
        solutions = []
        # for batch in range(1):
        for batch in range(len(problem.is_sat)):
            decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
            decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))

            def reify(phi):
                xs = list(zip(
                    [phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch + 1) * n_vars_per_batch)],
                    [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch + 1) * n_vars_per_batch)]
                ))

                def one_of(a, b): return (a and (not b)) or (b and (not a))

                assert (all([one_of(x[0], x[1]) for x in xs]))
                return [x[0] for x in xs]

            if self.solves(problem, batch, decode_cheap_A):
                solutions.append(reify(decode_cheap_A))
            elif self.solves(problem, batch, decode_cheap_B):
                solutions.append(reify(decode_cheap_B))
            else:
                L = np.reshape(final_lits.cpu().detach().numpy(), [2 * n_batches, n_vars_per_batch, self.args.dim])
                L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

                kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
                distances = kmeans.transform(L)
                scores = distances * distances

                def proj_vlit_flit(vlit):
                    if vlit < problem.n_vars:
                        return vlit - batch * n_vars_per_batch
                    else:
                        return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

                def decode_kmeans_A(vlit):
                    return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                           scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]

                decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                if self.solves(problem, batch, decode_kmeans_A):
                    solutions.append(reify(decode_kmeans_A))
                elif self.solves(problem, batch, decode_kmeans_B):
                    solutions.append(reify(decode_kmeans_B))
                else:
                    solutions.append(None)
        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                # print("[%d] %d" % (batch, vlit))
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True
