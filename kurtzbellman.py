import numpy as np
from scipy import stats

beta, rho, B, M = 0.5, 0.9, 10, 6

Z = stats.randint(0, B + 1)
sigma = np.empty(range(B + 1))
phi_z = Z.pmf(range(B + 1))
S = np.array(range(B + M + 1))


def U(x):
    return x**beta


def Gamma(x):
    return np.array(range(min(x, M) + 1))


def dist(v0, v1):
    return np.max(np.abs(v0 - v1))


def T(v):
    Tv = np.empty_like(S)
    sigma = np.empty_like(S)
    for x in S:
        feasible_vals = Gamma(x)
        max_val = -np.inf
        for a in feasible_vals:
            tmp_val = U(x - a) + rho * np.dot(v[a : a + B + 1], phi_z)
            if tmp_val > max_val:
                max_val = tmp_val
                sigma[x] = a
        Tv[x] = max_val
    return Tv, sigma


def value_iter(v0, tol=1e-5, max_iter=1e3, verbose=True):
    iter = 0
    success = False
    while True:
        v1, sigma = T(v0)
        d = dist(v0, v1)
        if verbose is True and iter % 10 == 0:
            print(f"Iteration: {iter}. Distance: {d}.")
        if d < tol:
            success = True
            break
        if iter > max_iter:
            break
        v0 = v1
        iter += 1
    return v1, sigma, success, iter, v0


res = value_iter(U(S))


# S = np.array(range(B + M + 1))
# Z = np.array(range(B + 1))
# tol = 0.00001
# max_iter = 1000

# # phi
# phi = stats.randint(0, B + 1)


# def U(x):
#     return x**beta


# def Gamma(x):
#     return np.array(range(0, min(x, M) + 1))


# def dist(v0, v1):
#     return max(np.abs(v0 - v1))


# def T(v):
#     Tv = np.empty_like(v)
#     x_iter = 0
#     for x in S:
#         possible_vals = Gamma(x)
#         iter = 0
#         for a in possible_vals:
#             vals = np.empty_like(possible_vals)
#             y = U(x - a)
#             for z in Z:
#                 y += rho * v[a + z] * phi.pmf(z)
#             vals[iter] = y
#             iter += 1
#         Tv[x_iter] = max(vals)
#         x_iter += 1
#     return Tv


# success = False
# c_iter = 0
# v0 = U(S)
# while True:
#     v1 = T(v0)
#     d = dist(v0, v1)
#     if d < tol:
#         success = True
#         break
#     c_iter += 1
#     if c_iter == max_iter:
#         break
#     v0 = v1
