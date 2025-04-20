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


# Sample sol
max_T = 12
T_vals = range(max_T)
profits = []
r = 0.05
rho = 1 / (1 + r)

psi = (0.2, 0.2, 0.6)
h = (1000, 0, -1000)
current_profits = np.inner(psi, h)
discount = rho
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
