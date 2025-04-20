#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 21:05:44 2025

@author: alex
"""

from ds import Ds
from random import uniform
import matplotlib.pyplot as plt
import numpy as np

def quadmap(x):
    return 4  * x * (1 - x)

q = Ds(quadmap, 0.2)

T1 = q.trajectory(100)
plt.plot(T1)
plt.show()
plt.close()

grid = np.linspace(2.7, 4, 50) 

res = list()
for c in grid:
    quad = lambda x: c * x * (1 - x)
    x = uniform(0, 1)
    q_tmp = Ds(quad, x)
    tmp_traj = q_tmp.trajectory(1000)
    tmp_traj = [[c] * 50, tmp_traj[(len(tmp_traj)-50):]]
    plt.scatter(tmp_traj[0], tmp_traj[1])
    

