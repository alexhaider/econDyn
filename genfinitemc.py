#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 21:46:41 2025

@author: alex
"""

from random import uniform

# import matplotlib.pyplot as plt
import numpy as np


# ------------
# Ex 4.2.15
# ------------
# recursive call to compute joint distribution
def path_prob(X, psi, p):
    len_x = len(X)
    if len_x == 1:
        return psi[X[0]]
    else:
        p_from = X[len_x - 2]
        p_to = X[len_x - 1]
        X.pop()
        return p[p_from][p_to] * path_prob(X, psi, p)


def sample(phi):
    u = uniform(0, 1)
    phi_cs = np.cumsum(phi)
    tmp = phi_cs[len(phi_cs) - 1]
    if tmp != 1:
        print(f"Standardizing input by dividing by {tmp}")
        phi_cs = phi_cs / tmp
        print(phi_cs)
    bucket = u <= phi_cs
    return np.where(bucket)[0][0]


class MC:
    def __init__(self, p, X):
        self.X = X
        self.p = p

    def update(self):
        self.X = sample(self.p[self.X])
        # self.X = choices([0, 1, 2], weights = self.p[self.X], k = 1)[0]

    def sample_path(self, n):
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path

    def marg_dist(self, t, n, psi):
        path_dist = list()
        for i in range(n):
            self.X = sample(psi)  # reset
            tmp = self.sample_path(t + 1)
            path_dist.append(tmp[len(tmp) - 1])
        res = []
        for i in range(len(self.p)):
            res.append(path_dist.count(i) / n)
        return res

    def marg_dist2(self, t, n, psi):
        res = dict()
        for i in range(len(psi)):
            res[i] = 0
        for i in range(n):
            self.X = sample(psi)  # reset
            tmp = self.sample_path(t + 1)
            fin_index = tmp.pop()
            res[fin_index] += 1
        for i in range(len(psi)):
            res[i] = res[i] / n
        return res

    def path_prob(self, psi, path):
        prob = psi[path[0]]
        for t in range(len(path) - 1):
            prob = prob * self.p[path[t]][path[t + 1]]
        return prob


# phi1 = [0.3, 0.2, 0.4, 0.1]

# res = list()
# for i in range(1000):
#     res.append(sample(phi1))

# counts, bins = np.histogram(res)
# plt.stairs(counts, bins)
