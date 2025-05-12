import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt

q = beta(5, 5)
obs = q.rvs(2000)
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, q.pdf(grid), "k-", linewidth=2)
plt.show()

q1 = stats.norm(0, 1)
obs1 = q1.rvs(2000)
grid1 = np.linspace(-4, 4, 1000)

fig1, ax1 = plt.subplots()
ax1.hist(obs1, bins=40, density=True)
ax1.plot(grid1, q1.pdf(grid1), "k--")


# ==============
# Root finding
# ==============


def bisect(f, a, b, tol=1e-11):
    lower, upper = a, b

    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        # print(f"eval at {middle}, value = {f(middle)}")
        if f(middle) > 0:
            lower, upper = lower, middle
        else:
            lower, upper = middle, upper

    return 0.5 * (upper + lower)


def f(x):
    return np.sin(4 * (x - 1 / 4)) + x + x**20 - 1


x = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
ax.plot(x, f(x), label="$f(x)$")
ax.axhline(ls="--", c="k")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.legend()
plt.show()

print(f"Root at {bisect(f, 0, 1)}.")

# scipy routines:
print(f"Bisection: {opt.bisect(f, 0, 1)}")
print(f"Newton-Raphson {opt.newton(f, 0.2)}")
print(f"Brent {opt.brentq(f, 0, 1)}")


# fixed point:
def g(x):
    return x**2


def g1(x):
    return g(x) - x


print(f"Fixed Point at {opt.fixed_point(g, 2)}")
print(f"Fixed Point at {opt.newton(g1, 1.1)}")


# ==============
# optimization
# ==============

opt.fminbound(g, -1, 2)
