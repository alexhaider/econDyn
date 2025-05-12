import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe


# a = np.zeros(3)
# type(a)

# b = np.array([[1, 2], [3, 4]])
# c = b[1, :]
# type(c)

# d = b.copy()


# -----------
# Exc 11.8.1
# -----------


def p(x, coef):
    X = np.ones_like(coef)
    X[1:] = x
    y = X.cumprod()
    return coef @ y


x = 2
coef = np.linspace(2, 4, 3)
print(p(x, coef))


# ------------------
# Exc 11.8.2
# Inverse Transform
# ------------------


class DiscreteRV:
    """
    Generates an array of draws from a discrete random variable with vector of probabilities given by q.
    """

    def __init__(self, q):
        self.q = q
        if min(self.q) < 0:
            print(f"\033[93mWarning: negative probability. Adding {min(self.q)}\033[0m")
            self.q += min(self.q)
        self.pmf = np.cumsum(q)
        mass_total = self.pmf[len(self.pmf) - 1]
        if mass_total != 1:
            print(
                f"\033[93mWarning: Probabiltiies do not sum to 1. Dividing by {mass_total}\033[0m"
            )
            self.pmf = self.pmf / mass_total

    def draw(self, k):
        samples = np.random.uniform(0, 1, k)
        return np.searchsorted(self.pmf, samples) + 1


q = np.array([-0.1, 0.3, 0.4, 0.3])
d = DiscreteRV(q)
print(d.draw(10))


# ------------------
# Exc 11.8.3
# ECDF
# ------------------


class ECDF:
    def __init__(self, sample):
        self.sample = sample
        self.len = len(sample)

    def __call__(self, x):
        return len(self.sample[self.sample < x]) / self.len

    def eval(self, x):
        return len(self.sample[self.sample < x]) / self.len

    def plot(self, a, b):
        grid_vals = np.linspace(a, b, 100)
        grid_ecdf = np.empty(len(grid_vals))
        for i, x in enumerate(grid_vals):
            grid_ecdf[i] = self.eval(x)
        plt.plot(grid_vals, grid_ecdf)
        plt.show()

    def plot2(self, a, b):
        grid_vals = np.linspace(a, b, 100)
        f = np.vectorize(self.__call__)
        plt.plot(grid_vals, f(grid_vals))
        plt.show()


samples = np.random.uniform(0, 1, 10000)
F = ECDF(samples)
print(f"ECDF: {F(0.5)}")
F.plot(0, 0.4)
F.plot2(0, 0.4)


samples = np.random.randn(10000)
F1 = ECDF(samples)
F1.plot(-2, 2)


# ------------------
# Exc 11.8.4
# ECDF
# ------------------

l = 100
np.random.seed(123)
x = np.random.randn(l, l)
y = np.random.randn(l)

qe.tic()
A = x / y
qe.toc()
# print(A)

qe.tic()
A1 = np.empty_like(A)
for i in range(l):
    for j in range(l):
        A1[i, j] = x[i, j] / y[j]
qe.toc()

# print(A1)
