#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 07:13:06 2025

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from ds import Ds

pH = ((0.971, 0.029, 0.000),
      (0.145, 0.778, 0.077),
      (0.000, 0.508, 0.492))

psi = (0.3, 0.4, 0.3)

M = lambda phi: np.dot(phi, pH)
markovds = Ds(h=M, x=psi)
T = markovds.trajectory(100)

# return
h = (1000, 0, -1000)
res = np.dot(T[len(T)-1], h)
print(res)

plt.plot(T)
plt.show()


# starting in NG
delta_x1 = (1,0,0)
m1 = Ds(h=M, x = delta_x1)
T1 = m1.trajectory(5)
res1 = np.dot(T1[len(T1)-1], h)

# starting in SR
delta_x2 = (0,0,1)
m2 = Ds(h=M, x = delta_x2)
T2 = m2.trajectory(5)
res2 = np.dot(T2[len(T2)-1], h)

print(res1)
print(res2)
