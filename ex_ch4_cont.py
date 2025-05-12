import numpy as np
import random
from genfinitemc import MC, sample

# ==============
# Ex 4.3.25
# ==============


def d1(v0, v1):
    d1 = 0
    for i in range(len(v0)):
        d1 += np.abs(v0[i] - v1[i])
    return d1


def fp(M, phi, tol, max_iter):
    v0 = phi
    v1 = phi @ M
    dist = d1(v0, v1)
    iter = 0
    success = False
    while dist > tol and iter < max_iter:
        iter += 1
        v0 = v1
        v1 = v1 @ M
        dist = d1(v0, v1)
    if dist < tol:
        success = True
    return (v0, v1, dist, iter, success)


pQ = np.array(
    [
        [0.97, 0.03, 0, 0, 0],
        [0.05, 0.92, 0.03, 0, 0],
        [0, 0.04, 0.92, 0.04, 0],
        [0, 0, 0.04, 0.94, 0.02],
        [0, 0, 0, 0.01, 0.99],
    ]
)
psi = np.array([1, 0, 0, 0, 0])
res = fp(pQ, psi, 1e-6, 10000)
print(res)


# ==============
# Ex 4.3.27
# ==============

# evaluate analytically
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))
x = [0, 0, 1]
res = fp(pH, x, 1e-6, 10000)  # find the fixed point
expected_t = [1 / i for i in res[0]]

# now by sim
pH = np.array([[0.971, 0.029, 0.000], [0.145, 0.778, 0.077], [0.000, 0.508, 0.492]])
x = np.array([0, 1, 0])
x_start = np.where(x == 1)[0][0]

n = 10_000
ret_time = np.empty(n)
for i in range(n):
    iter = 0
    while True:
        iter += 1
        x = x @ pH
        x_smpl = random.choices(range(len(x)), weights=x, k=1)[0]
        if x_smpl == x_start:
            break
    ret_time[i] = iter

print(f"Ret time: {np.mean(ret_time)}")


# ==============
# Ex 4.3.28
# ==============


def demand(d):
    return 0.5 ** (d + 1)


Q = 5
q = 2
Mq = np.empty((Q + 1, Q + 1))

for i in range(Q + 1):
    for j in range(Q + 1):
        if j == 0 and i <= q:
            Mq[i, j] = 1
            for k in range(Q):
                Mq[i, j] = Mq[i, j] - demand(k)
        if j == 0 and i > q:
            Mq[i, j] = 1
            for k in range(i):
                Mq[i, j] = Mq[i, j] - demand(k)
        if j > 0 and i <= q:
            Mq[i, j] = demand(Q - j)
        if j > 0 and i > q:
            if i < j:
                Mq[i, j] = 0
            else:
                Mq[i, j] = demand(i - j)


x = np.array([1, 0, 0, 0, 0, 0])
fp_r, *rest = fp(Mq, x, 1e-10, 2000)
print(f"Stationary distribution: {fp_r}")


# ==============
# Ex 4.3.28
# ==============
def d1(v0, v1):
    d1 = 0
    for i in range(len(v0)):
        d1 += np.abs(v0[i] - v1[i])
    return d1


def fp(M, phi, tol, max_iter):
    v0 = phi
    v1 = phi @ M
    dist = d1(v0, v1)
    iter = 0
    success = False
    while dist > tol and iter < max_iter:
        iter += 1
        v0 = v1
        v1 = v1 @ M
        dist = d1(v0, v1)
    if dist < tol:
        success = True
    return (v0, v1, dist, iter, success)


def demand(d):
    return 0.5 ** (d + 1)


def profit(C, d, x):
    return min(x, d) - C


def create_Mq(Q, q):
    Mq = np.empty((Q + 1, Q + 1))
    for i in range(Q + 1):
        for j in range(Q + 1):
            if j == 0 and i <= q:
                Mq[i, j] = 1
                for k in range(Q):
                    Mq[i, j] = Mq[i, j] - demand(k)
            if j == 0 and i > q:
                Mq[i, j] = 1
                for k in range(i):
                    Mq[i, j] = Mq[i, j] - demand(k)
            if j > 0 and i <= q:
                Mq[i, j] = demand(Q - j)
            if j > 0 and i > q:
                if i < j:
                    Mq[i, j] = 0
                else:
                    Mq[i, j] = demand(i - j)
    return Mq


Q = 7
# q = 2
C = 0.1

x_start = np.zeros(Q + 1)
x_start[0] = 1

all_pi = np.zeros(Q + 1)
for q in range(Q + 1):
    tmp_M = create_Mq(Q, q)
    fp_r, *rest = fp(tmp_M, x_start, 1e-10, 2000)
    pi = np.zeros(Q + 1)
    for x in range(Q + 1):
        for d in range(Q + 1):
            if x < q:
                pi[x] += profit(C, d, q) * tmp_M[x, d]
            else:
                pi[x] += profit(0, d, x) * tmp_M[x, d]
        pi[x] = pi[x] * fp_r[x]
    all_pi[q] = sum(pi)

print(f"Optimal q: {np.where(all_pi == max(all_pi))[0][0]}")


# ==================
# Example 4.3.32
# ==================

r_n = np.random.randn(int(1e6))
r_n2 = [i**2 for i in r_n]
print(np.mean(r_n2))


# ==================
# Exercise 4.3.34
# ==================

n = int(1e6)
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))

psi = (0.3, 0.4, 0.3)

h = MC(pH, sample(psi))
T1 = h.sample_path(n)

buckets = np.zeros_like(psi)
for i in range(n):
    buckets[T1[i]] += 1
buckets = [i / n for i in buckets]


# ==================
# Exercise 4.3.36
# ==================
h1 = (1000, 0, -1000)

n = int(1e6)
pH = ((0.971, 0.029, 0.000), (0.145, 0.778, 0.077), (0.000, 0.508, 0.492))

psi = (0.3, 0.4, 0.3)

h = MC(pH, sample(psi))
T1 = h.sample_path(n)

profit = 0
for i in range(n):
    profit += h1[T1[i]]
print(f"Expected profit = {profit / n} ")
