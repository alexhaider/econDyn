#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:12:50 2025

@author: alex
"""

# from genfinitemc import sample, MC
import matplotlib.pyplot as plt
import numpy as np

pQ = (
    (0.97, 0.03, 0, 0, 0),
    (0.05, 0.92, 0.03, 0, 0),
    (0, 0.04, 0.92, 0.04, 0),
    (0, 0, 0.04, 0.94, 0.02),
    (0, 0, 0, 0.01, 0.99),
)

x = (1, 0, 0, 0, 0)

res = list()
for t in range(161):
    res.append(x)
    x = np.matmul(x, pQ)

x_axis = ["x" + str(i + 1) for i in range(len(x))]
t = (10, 60, 160)
for i in t:
    plt.bar(x_axis, res[i])
    plt.show()

x = (0, 0, 0, 0, 1)

res = list()
for t in range(161):
    res.append(x)
    x = np.matmul(x, pQ)

x_axis = ["x" + str(i + 1) for i in range(len(x))]
t = (10, 60, 160)
for i in t:
    plt.bar(x_axis, res[i], color="orange")
    plt.show()
