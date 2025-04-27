import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


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
