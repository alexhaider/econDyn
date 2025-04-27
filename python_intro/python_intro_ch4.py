import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, generator_type):
    if generator_type not in ("norm", "uniform"):
        print("invalid input")
        return None
    if generator_type == "norm":
        f = np.random.standard_normal
    else:
        f = np.random.uniform
    vals = np.empty(n)
    for i in range(n):
        vals[i] = f()
    return vals


abc = generate_data(100, "abc")
norm = generate_data(100, "norm")
unif = generate_data(100, "uniform")

plt.plot(norm)
plt.plot(unif)
plt.show()


def gen_data(n, generator_type):
    vals = list()
    for i in range(n):
        vals.append(generator_type())
    return vals


vals = gen_data(100, np.random.uniform)  # passing a function


# ================
# Recursive Calls
# ================


def x(t):
    if t == 0:
        return 1
    else:
        return 2 * x(t - 1)


x(23)
x(3)

# Factorial


def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)


factorial(4)
factorial(3)


def rand_bin(n, p):
    u = np.random.uniform(0, 1, n)

    ret_val = np.empty(n)
    for i in range(n):
        if u[i] < p:
            ret_val[i] = 1
        else:
            ret_val[i] = 0
    return ret_val


print(sum(rand_bin(10_000, 0.3)) / 10_000)


# Ex 4.6.3


def coin_flip(n, k):
    u = np.random.uniform(0, 1, n)
    count = 0
    payoff = 0
    for i in range(n):
        if u[i] > 0.5:
            count += 1
        else:
            count = 0
        if count == k:
            payoff = 1
            break
    return payoff, u


pay, flips = coin_flip(10, 2)


# =======================
# Advanced ex (Recursion)
# =======================


def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


""""
                     fib(4)
                   /       \
              fib(3)       fib(2)
             /     \       /     \
         fib(2)   fib(1)  fib(1)  fib(0)
        /     \
    fib(1)   fib(0)
"""

print(fib(4))
