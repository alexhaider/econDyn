#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:14:06 2025

@author: alex
"""

# ------------
# Ex 4.2.12
# ------------

import numpy as np
import pandas as pd
import math
from genfinitemc import MC, path_prob, sample
import matplotlib.pyplot as plt
import itertools

# Normal Growth start
delta_x = (1, 0, 0)
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
h = (1000, 0, -1000)

ret = delta_x
for i in range(5):
    ret = np.dot(ret, pH)
ret = np.dot(ret, h)

print(f"Return with {delta_x} is equal to {round(ret, 2)}")

# Severe recession start
delta_x = (0, 0, 1)

ret = delta_x
for i in range(5):
    ret = np.dot(ret, pH)
ret = np.dot(ret, h)

print(f"Return with {delta_x} is equal to {round(ret, 2)}")


# ------------
# Ex 4.2.13
# ------------

t = 1000
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
h = (1000, 0, -1000)

ret_1e3 = list()
for i in range(len(h)):
    psi = [0, 0, 0]
    psi[i] = 1
    for j in range(t):
        psi = np.dot(psi, pH)
    ret_1e3.append(np.dot(psi, h))

# ------------
# Ex 4.2.14
# ------------

t = 5
psi = (0.2, 0.2, 0.6)
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
h = (1000, 0, -1000)

for j in range(t):
    psi = np.dot(psi, pH)
ret2 = np.dot(psi, h)

# ------------
# Ex 4.2.14
# ------------
psi = (0.2, 0.2, 0.6)
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
path = (0, 1, 0)
X = 0

mc = MC(pH, X)
print(mc.path_prob(psi, path))


# ------------
# Ex 4.2.15
# ------------
x = [0, 1, 0]
print(f"Recursive compute: {path_prob(x, psi, pH)}")


# ------------
# Ex 4.2.18
# ------------

# monte carlo sim (Ex 4. 2. 19)
t = int(1e5)

counter = 0
for t in range(t):
    mc1 = MC(pH, sample(psi))
    tmp = mc1.sample_path(3)
    if 0 not in tmp:
        counter += 1
print(f"by sim: {counter / t}")

# analytic sol: manual
prob = (
    psi[1] * pH[1][1] * pH[1][1]
    + psi[1] * pH[1][1] * pH[1][2]
    + psi[1] * pH[1][2] * pH[2][1]
    + psi[1] * pH[1][2] * pH[2][2]
    + psi[2] * pH[2][2] * pH[2][2]
    + psi[2] * pH[2][2] * pH[2][1]
    + psi[2] * pH[2][1] * pH[1][2]
    + psi[2] * pH[2][1] * pH[1][1]
)
print(prob)

prob1 = psi[1] * (pH[1][1] * sum(pH[1][1:3]) + pH[1][2] * sum(pH[2][1:3])) + psi[2] * (
    pH[2][1] * sum(pH[1][1:3]) + pH[2][2] * sum(pH[2][1:3])
)
print(prob1)

# with path_prob
res = 0
rec_states = 1, 2
for x1 in rec_states:
    for x2 in rec_states:
        for x3 in rec_states:
            x = [x1, x2, x3]
            res += path_prob(x, psi, pH)

print(res)

# ------------
# Ex 4.2.20
# ------------
res = 0
h = (1000, 0, -1000)
r = 0.05
rho = 1 / (1 + r)

for x0 in range(3):
    for x1 in range(3):
        for x2 in range(3):
            x = [x0, x1, x2]
            pay = rho**0 * h[x0] + rho**1 * h[x1] + rho**2 * h[x2]
            res = res + pay * path_prob(x, psi, pH)

print(f"NPV: {res}")


# Parameters
r = 0.05
rho = 1 / (1 + r)

# Transition matrix
P = np.array([[0.971, 0.029, 0.000], [0.145, 0.778, 0.077], [0.000, 0.508, 0.492]])

# Initial distribution
psi = np.array([0.2, 0.2, 0.6])

# Reward function
h = np.array([1000, 0, -1000])

# All paths of length 3 (x0, x1, x2)
states = [0, 1, 2]
total = 0

for x0, x1, x2 in itertools.product(states, repeat=3):
    prob = psi[x0] * P[x0, x1] * P[x1, x2]
    reward = rho**0 * h[x0] + rho**1 * h[x1] + rho**2 * h[x2]
    total += prob * reward

print(f"Expected discounted reward: {total:.2f}")

# ------------
# Ex 4.2.21
# ------------
# via Monte Carlo

runs = 10_000
pay = list()

for t in range(runs):
    mc2 = MC(pH, sample(psi))
    traj = mc2.sample_path(3)
    pay.append(
        h[traj[0]] + (1 / (1 + r)) * h[traj[1]] + (1 / (1 + r)) ** 2 * h[traj[2]]
    )
print(f"NPV sim: {sum(pay) / runs}")


# Exercse 4.23

psi = [0.2, 0.2, 0.6]
pH = ((0.971, 0.029, 0), (0.145, 0.778, 0.077), (0, 0.508, 0.492))
h = (1000, 0, -1000)

T = 3
r = 0.05
rho = 1 / (1 + r)

profit = 0
for i in range(T):
    if i > 0:
        psi = np.dot(psi, pH)
    profit += rho**i * np.dot(psi, h)
print(f"Profit: {profit}")

# break even
psi = [0.2, 0.2, 0.6]
t = 0
profit = 0
while True:
    if t > 0:
        psi = np.dot(psi, pH)
    profit += rho**t * np.dot(psi, h)
    if profit > 0:
        break
    t += 1
print(f"Break even: {t}")


# plotting
T = 12
r = 0.05
rho = 1 / (1 + r)

psi = [0.2, 0.2, 0.6]
h = (1000, 0, -1000)

profit = 0
all_profits = list()
for i in range(T):
    if i > 0:
        psi = np.dot(psi, pH)
    profit += rho**i * np.dot(psi, h)
    all_profits.append(profit)
print(f"Profit: {profit}")

ts = np.linspace(0, T - 1, T)
plt.plot(ts, all_profits, label="Profits")
plt.plot(ts, np.zeros(T), "--", label="breakeven")
plt.legend()
plt.show()

# Sample sol
max_T = 12
T_vals = range(max_T)
profits = []
r = 0.05
rho = 1 / (1 + r)

psi = (0.2, 0.2, 0.6)
h = (1000, 0, -1000)
current_profits = np.inner(psi, h)
discount = 1
Q = np.identity(3)

for t in T_vals:
    Q = np.dot(Q, pH)
    discount = discount * rho
    current_profits += discount * np.inner(psi, np.dot(Q, h))
    profits.append(current_profits)

fig, ax = plt.subplots()
ax.plot(profits, label="profits")
ax.plot(np.zeros(max_T), "--", label="break even")
ax.set_xlabel("time")
ax.legend()

plt.show()


# --------------
# Listing 4.6
# --------------
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
size_pH = len(pH)

I = np.identity(size_pH)
M_one = np.ones((size_pH, size_pH))
v_one = np.ones((size_pH, 1))

A = np.transpose(I - pH + M_one)
x = np.linalg.solve(A, v_one)
print(f"Stationary dist: {x}")

x1 = np.linalg.inv(A) @ v_one
print(f"Stationary dist: {x1}")


# --------------
# Ex 4.3.8
# --------------
pQ = (
    (0.97, 0.03, 0, 0, 0),
    (0.05, 0.92, 0.03, 0, 0),
    (0, 0.04, 0.92, 0.04, 0),
    (0, 0, 0.04, 0.94, 0.02),
    (0, 0, 0, 0.01, 0.99),
)
psi = (0.1, 0.2, 0.1, 0.4, 0.2)


def find_stat_dist(p):
    size_p = len(p)
    eye = np.identity(size_p)
    M_1 = np.ones((size_p, size_p))
    v_1 = np.ones((size_p, 1))

    A = np.transpose(eye - p + M_1)
    return np.linalg.solve(A, v_1).flatten()


sdist = find_stat_dist(pQ)

x = (1, 0, 0, 0, 0)

res = list()
for t in range(1001):
    res.append(x)
    x = np.matmul(x, pQ)

mc_1 = MC(pQ, x)
pth = mc_1.marg_dist(1000, 1000, x)

x_axis = ["x" + str(i + 1) for i in range(len(x))]
t = (10, 500, 1000)
cols = ("blue", "orange", "yellow")
for i, time in enumerate(t):
    plt.bar(x_axis, res[time], color=cols[i])
plt.bar(x_axis, sdist, color="green")
plt.bar(x_axis, pth, color="brown")
plt.show()


# --------------
# Ex 4.3.9
# --------------
sdist_ph = find_stat_dist(pH)
print(np.inner(sdist_ph, h))

# --------------
# Ex 4.3.10
# --------------
psi = sdist
for i in range(20):
    psi = psi @ pQ


# --------------
# Ex 4.3.11
# --------------
p1 = np.array([[0, 1], [1, 0]])
print(find_stat_dist(p1))


all_psi = np.zeros((100, 2))
tmp = np.array([0.8, 0.2])
for i in range(100):
    all_psi[i,] = tmp
    tmp = tmp @ p1


# -----------------------------
# Dobrushin coeffe of p_Q^23
# -----------------------------


pQ = np.array(
    [
        [0.97, 0.03, 0, 0, 0],
        [0.05, 0.92, 0.03, 0, 0],
        [0, 0.04, 0.92, 0.04, 0],
        [0, 0, 0.04, 0.94, 0.02],
        [0, 0, 0, 0.01, 0.99],
    ]
)

pQ23 = np.identity(len(pQ))
for i in range(25):
    pQ23 = pQ23 @ pQ


def get_alpha(Mat):
    trans = np.zeros(sum(range(len(Mat) + 1)))
    index = 0
    for i in range(len(Mat)):
        for j in range(i, len(Mat)):
            for k in range(len(Mat)):
                trans[index] += min(Mat[i, k], Mat[j, k])
            index += 1
    return min(trans)


# print(f"The Dobrushin coefficient is: {min(trans)}.")


def test_stab(Mat, n=50):
    m = np.identity(len(Mat))
    success = False
    for i in range(n):
        m = m @ Mat
        alpha = get_alpha(m)
        if alpha > 0:
            success = True
            break
    return (success, i)


pq_test = test_stab(pQ)
