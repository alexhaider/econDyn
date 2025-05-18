import numpy as np
import matplotlib.pyplot as plt
from numba import jit

beta, rho = 0.5, 0.9
z_bar, s_bar = 10, 5

S = np.arange(z_bar + s_bar + 1)  # State space = 0,...,z_bar + s_bar
Z = np.arange(z_bar + 1)  # Shock space = 0,...,z_bar


def U(c):
    "Utility function."
    return c**beta


def phi(z):
    "Probability mass function, uniform distribution."
    return 1.0 / len(Z) if 0 <= z <= z_bar else 0


def Gamma(x):
    "The correspondence of feasible actions."
    return range(min(x, s_bar) + 1)


def T(v):
    Tv = np.empty_like(v)
    for x in S:
        # Compute the value of the objective function for each
        # a in Gamma(x) and record highest value.
        running_max = -np.inf
        for a in Gamma(x):
            y = U(x - a) + rho * sum(v[a + z] * phi(z) for z in Z)
            if y > running_max:
                running_max = y
        # Store the maximum reward for this x in Tv
        Tv[x] = running_max
    return Tv


def get_greedy(w):
    sigma = np.empty_like(w)
    for x in S:
        running_max = -np.inf
        for a in Gamma(x):
            y = U(x - a) + rho * sum(w[a + z] * phi(z) for z in Z)
            # Record the action that gives highest value
            if y > running_max:
                running_max = y
                sigma[x] = a
    return sigma


def compute_value_function(tol=1e-4, max_iter=1000, verbose=True, print_skip=5):
    # Set up loop
    v = [U(x) for x in S]  # Initial condition
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_new = T(v)
        error = np.max(np.abs(v - v_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")
        v = v_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return v_new


v_star = compute_value_function()

fig, ax = plt.subplots()
ax.plot(S, v_star, "k-", label="approximate value function")
ax.legend()
ax.set_xlabel("$x$")
ax.set_ylabel("value")

# plt.savefig("vfiv.pdf")

plt.show()


sigma_star = get_greedy(v_star)

fig, ax = plt.subplots()
ax.plot(S, sigma_star, "k-", label="optimal policy")
ax.legend()

plt.show()


p_sigma = np.empty((len(S), len(S)))

for x in S:
    for y in S:
        p_sigma[x, y] = phi(y - sigma_star[x])
