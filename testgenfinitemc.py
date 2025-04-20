#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:39:31 2025

@author: alex
"""

from genfinitemc import sample, MC
import matplotlib.pyplot as plt

pH = ((0.971, 0.029, 0.000),
      (0.145, 0.778, 0.077),
      (0.000, 0.508, 0.492))

psi = (0.3, 0.4, 0.4)

h = MC(pH, sample(psi))
T1 = h.sample_path(1000)

plt.plot(T1)
plt.show()
plt.close()

psi2 = (0.8, 0.1, 0.1)
h.X = sample(psi2)
T2 = h.sample_path(1000)

plt.plot(T2)
plt.show()

# Hamilton chain (with pH) marginal dist (exercise 4.2.3)
psi = (0, 0, 1)
t = 10

h = MC(pH, sample(psi))
res = h.marg_dist2(t, 10_000, psi)

