import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# --------------------------------

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y, "r-", linewidth=2, label="$y=\sin(x)$", alpha=0.6)
ax.legend()
ax.set_yticks(np.linspace(-1, 1, 5))
plt.show()

# --------------------------------
fig, ax = plt.subplots()
x = np.linspace(-4, 4, 100)
for i in range(3):
    m, s = np.random.uniform(-1, 1), np.random.uniform(1, 2)
    y = norm.pdf(x, loc=m, scale=s)
    current_label = f"$\mu = {m:.2}, \sigma = {s: .2}$"
    ax.plot(x, y, linewidth=2, alpha=0.6, label=current_label)
ax.legend()
plt.show()

# --------------------------------
num_rows, num_cols = 3, 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))
for i in range(num_rows):
    for j in range(num_cols):
        m, s = np.random.uniform(-1, 1), np.random.uniform(1, 2)
        x = norm.rvs(loc=m, scale=s, size=100)
        axes[i, j].hist(x, alpha=0.6, bins=20)
        t = f"$\mu = {m:.2}, \sigma = {s: .2}$"
        axes[i, j].set(title=t, xticks=np.linspace(-4, 4, 9), yticks=[])
plt.show()

# -------------------
# Ex 12.5.1
# -------------------


def f(x, theta):
    return np.cos(np.pi * theta * x) * np.exp(-x)


fig, ax = plt.subplots()

x_vals = np.linspace(0, 5, 100)
theta_vals = np.linspace(0, 2, 10)
for i in theta_vals:
    ax.plot(x_vals, f(x_vals, i), label=f"$\\theta = {i: .2}$")
    ax.legend(loc="upper right")
plt.show()


# ===================
# Testing
# ===================


fig, axes = plt.subplots(2, 2)

norm_data = np.empty((1000, 4))
for i in range(4):
    m, s = np.random.uniform(-1, 1), np.random.uniform(1, 5)
    norm_data[:, i] = np.random.randn(1000) * s + m
axes[0, 0].hist(norm_data[:, 0], alpha=0.6, bins=20)
axes[0, 1].hist(norm_data[:, 1], alpha=0.6, bins=20)
axes[1, 0].hist(norm_data[:, 2], alpha=0.6, bins=20)
axes[1, 1].hist(norm_data[:, 3], alpha=0.6, bins=20)
plt.show()
