import numpy as np
import matplotlib.pyplot as plt


# white noise process 1

eps = np.random.randn(100)
plt.plot(eps)
plt.show()

# white noise process 2

ts_length = 100
eps_vals = list()

for i in range(ts_length):
    e = np.random.randn()
    eps_vals.append(e)

plt.plot(eps_vals)
plt.show()

# bank account
r = 0.025
T = 50
b = np.empty(T + 1)
b[0] = 10

for t in range(T):
    b[t + 1] = b[t] * (1 + r)

plt.plot(b, label="bank balance")
plt.legend()
plt.show()


# Ex 3.6.1
T = 200
alpha = 0.9
ts = np.zeros(T + 1)
eps = np.random.randn(T)

for i in range(T):
    ts[i + 1] = alpha * ts[i] + eps[i]

plt.plot(ts)
plt.show()

# Ex 3.6.2
T = 200
a_series = [0, 0.8, 0.98]
ts = np.zeros((len(a_series), T + 1))

for i in range(len(a_series)):
    a = a_series[i]
    for t in range(T):
        ts[i, t + 1] = a * ts[i, t] + np.random.randn()

for i in range(len(a_series)):
    a = a_series[i]
    plt.plot(ts[i,], label=r"$\alpha = $" + str(a))

plt.legend()
plt.show()


# Ex 3.6.3
T = 200
alpha = 0.9
ts = np.zeros(T + 1)
eps = np.random.randn(T)

for i in range(T):
    ts[i + 1] = alpha * np.abs(ts[i]) + eps[i]

plt.plot(ts)
plt.show()


# Ex 3.6.4
T = 200
alpha = 0.9
ts = np.zeros(T + 1)
eps = np.random.randn(T)

for i in range(T):
    val = -ts[i] if ts[i] < 0 else ts[i]
    ts[i + 1] = alpha * val + eps[i]

plt.plot(ts)
plt.show()

# Ex 3.6.5

T = 10_000
# us = np.empty((T, 2))
counter = 0
for i in range(T):
    u = np.random.uniform(0, 1, 2)
    # us[i,] = u
    dist = np.sqrt(u[0] ** 2 + u[1] ** 2)
    if dist <= 1:
        counter += 1

print(f"Pi is approx. {4 * counter / T}")
