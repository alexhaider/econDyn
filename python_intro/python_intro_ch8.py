import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from random import uniform

# ---------
# Consumer
# ---------


class Consumer:
    def __init__(self, w):
        "Initialize the consumer with w USD of wealth"
        self.wealth = w

    def earn(self, y):
        "The consumer earns y USD"
        self.wealth += y

    def spend(self, x):
        "The consumer spends x USD if feasible"
        new_wealth = self.wealth - x
        if new_wealth < 0:
            print("Insufficient funds")
        else:
            self.wealth = new_wealth


consumers = list()
for i in range(100):
    consumers.append(Consumer(np.random.uniform(10, 100)))

c1 = Consumer(10)


# ---------
# Solow
# ---------


class Solow:
    def __init__(self, n=0.05, s=0.25, delta=0.1, alpha=0.3, z=2.0, k=1.0):
        self.n, self.s, self.delta, self.alpha, self.z = n, s, delta, alpha, z
        self.k = k

    def unpack_params(self):
        n, s, delta, alpha, z = self.n, self.s, self.delta, self.alpha, self.z
        return n, s, delta, alpha, z

    def h(self):
        n, s, delta, alpha, z = self.unpack_params()
        return (s * z * self.k**alpha + (1 - delta) * self.k) / (1 + n)

    def update(self):
        self.k = self.h()

    def steady_state(self):
        n, s, delta, alpha, z = self.unpack_params()
        return (s * z / (n + delta)) ** (1 / (1 - alpha))

    def generate_sequence(self, t):
        path = []
        for i in range(t):
            path.append(self.k)
            self.update()
        return path


s1 = Solow()
s2 = Solow(k=8.0)

T = 60
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot([s1.steady_state()] * T, "k-", label="steady state")

for s in s1, s2:
    lb = f"capital series from initial condition {s.k}"
    ax.plot(s.generate_sequence(T), "o-", lw=2, alpha=0.6, label=lb)

ax.set_xlabel("$t$", fontsize=14)
ax.set_ylabel("$k_t$", fontsize=14)
ax.legend()
plt.show()


class Market:
    def __init__(self, ad, bp, az, bz, tax):
        self.ad, self.bd, self.az, self.bz, self.tax = ad, bp, az, bz, tax
        if ad < az:
            raise ValueError("Insufficient demand.")

    def unpack_params(self):
        ad, bp, az, bz, tax = self.ad, self.bp, self.az, self.bz, self.tax
        return ad, bp, az, bz, tax

    def price(self):
        return (self.ad - self.az + self.bz * self.tax) / (self.bz + self.bd)

    def quantity(self):
        return self.ad - self.bd * self.price()

    def consumer_surp(self):
        def integrand(x):
            return self.ad / self.bd - x / self.bd

        area, error = quad(integrand, 0, self.quantity())
        return area - self.price() * self.quantity()

    def cs(self):
        def area_below(x):
            return self.price()

        area, error = quad(self.inverse_demand, 0, self.quantity())
        area2, error2 = quad(area_below, 0, self.quantity())
        return area - area2

    def producer_surp(self):
        def integrand(x):
            return -self.az / self.bz + x / self.bz

        area, error = quad(integrand, 0, self.quantity())
        return (self.price() - self.tax) * self.quantity() - area

    def ps(self):
        def area_above(x):
            return self.price() - self.tax

        area, error = quad(self.inverse_supply_no_tax, 0, self.quantity())
        area2, error2 = quad(area_above, 0, self.quantity())
        return area2 - area

    def taxrev(self):
        return self.tax * self.quantity()

    def inverse_demand(self, x):
        return self.ad / self.bd - x / self.bd

    def inverse_supply(self, x):
        return -(self.az / self.bd) + x / self.bz * x + self.tax

    def inverse_supply_no_tax(self, x):
        return -(self.az / self.bd) + x / self.bz


baseline_params = 15, 0.5, -2, 0.5, 3
m = Market(*baseline_params)
print(f"Price: {m.price()}")
print(f"Consumer surplus: {m.consumer_surp()}")
print(f"Consumer surplus2: {m.cs()}")

print(f"Producer surplus: {m.producer_surp()}")
print(f"Prod: {m.ps()}")


# class Chaos:
#     def __init__(self, x0, r):
#         self.x = x0
#         self.r = r

#     def update(self):
#         self.x = self.r * self.x * (1 - self.x)

#     def generate_sequence(self, n):
#         path = list()
#         for i in range(n):
#             path.append(self.x)
#             self.update()
#         return path

#     def __call__(self):
#         print(f"Current state: {self.x}")


# ch = Chaos(0.1, 4.0)
# ch.generate_sequence(5)

# # Bifurcation plot

# fig, ax = plt.subplots()
# rs = np.linspace(2.5, 4, 100)
# for i in rs:
#     tmp_sys = Chaos(0.5, i)
#     tmp_res = tmp_sys.generate_sequence(1000)[-50:]
#     ax.plot([i] * len(tmp_res), tmp_res, "b.", ms=0.6)

# plt.show()


# ---------
# Exc 8.5.1
# ---------


class ECDF:
    def __init__(self, sample):
        self.sample = sample

    def __call__(self, x):
        return sum([i <= x for i in self.sample]) / len(self.sample)


samples = [uniform(0, 1) for i in range(1000)]
F = ECDF(samples)
print(F(0.5))

# ---------
# Exc 8.5.2
# ---------


class Poly:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        res = 0.0
        for index, coeff in enumerate(self.a):
            res += coeff * x**index
        return res

    def differentiate(self):
        new_coeffs = []
        for index, coeff in enumerate(self.a):
            new_coeffs.append(index * coeff)
        del new_coeffs[0]
        return new_coeffs
