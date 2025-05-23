import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ==============
# Ex 5.1.2
# ==============

beta, rho, B, M = 0.5, 0.9, 10, 5

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
    Tv = np.empty(len(S))
    sigma = np.empty(len(S))
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


def value_iter(v0, tol=1e-4, max_iter=1e3, verbose=True):
    iter = 0
    while True:
        v1, sigma = T(v0)
        d = dist(v0, v1)
        if verbose is True and iter % 10 == 0:
            print(f"Iteration: {iter}. Distance: {d}.")
        if d < tol:
            success = True
            break
        if iter > max_iter:
            success = False
            break
        v0 = v1
        iter += 1
    return v1, sigma, success, iter, v0


v_star1, sigma, *rest = value_iter(U(S))
fig, ax = plt.subplots()
ax.plot(S, v_star1, label="Value function iteration\n(approx. value function)")
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("$v^*$")
plt.show()
sigma_vfi = sigma

# ==============
# Ex 5.1.3
# ==============


def get_alpha(Mat):
    trans = np.zeros(sum(range(len(Mat) + 1)))
    index = 0
    for i in range(len(Mat)):  # x loop
        for j in range(i, len(Mat)):  # x' loop
            for k in range(len(Mat)):  # y loop
                trans[index] += min(Mat[i, k], Mat[j, k])
            index += 1
    return min(trans)


def compute_stat_dist(Mat):
    size_mat = len(Mat)
    eye = np.identity(size_mat)
    M_one = np.ones((size_mat, size_mat))
    v_one = np.ones((size_mat, 1))

    A = np.transpose(eye - pB + M_one)
    x = np.linalg.solve(A, v_one)
    return x.flatten()


pB = np.empty((B + M + 1, B + M + 1))

for i in range(B + M + 1):
    for j in range(B + M + 1):
        a = sigma[i]
        # X_{t+1} = a_t + W_{t+1}
        # it's therefore not possible for X_{t+1} < a_t
        # it's also not possible to have X_{t+1} > a_t + max(W_{t+1})
        # otherwise: probability is fixed
        if j < a or j > a + B:
            pB[i, j] = 0
        else:
            pB[i, j] = phi_z[0]

print(f"Dobrushin coeff: {get_alpha(pB)}.")

x_stat = compute_stat_dist(pB)
fig, ax = plt.subplots()
ax.bar(S, x_stat, label=r"$\psi^\ast$")
ax.legend()
ax.set_xlabel("x")

plt.show()


# =======================================================
# Policy iteration
# =======================================================
beta, rho, B, M = 0.5, 0.9, 10, 5

Z = stats.randint(0, B + 1)
sigma = np.empty(range(B + 1))
phi_z = Z.pmf(range(B + 1))
S = np.array(range(B + M + 1))


def get_p(sigma, size=B + M + 1):
    pB = np.empty((size, size))

    for i in range(size):
        for j in range(size):
            a = sigma[i]
            if j < a or j > a + B:
                pB[i, j] = 0
            else:
                pB[i, j] = phi_z[0]
    return pB


def get_v_sigma(sigma, iter_v=50):
    N = len(S)
    p_mat = get_p(sigma)

    def M_sigma(h):
        return p_mat @ r_sigma

    discount = 1
    v_sigma = np.zeros(N)
    r_sigma = U(S - sigma)
    for i in range(iter_v):
        v_sigma += discount * r_sigma
        discount = discount * rho
        r_sigma = M_sigma(r_sigma)
    return v_sigma


def policy_iter(sig=np.zeros(len(S)), max_iter=1000, tol=1e-5):
    iter = 0
    while True:
        v_sig = get_v_sigma(sig)
        Tv, sig_star = T(v_sig)
        e = sig - sig_star
        if np.max(np.abs(e)) < tol:
            success = True
            break
        if iter == max_iter:
            success = False
            break
        sig = sig_star
        iter += 1
    return sig_star, success


sigma_pi, *rest = policy_iter()
