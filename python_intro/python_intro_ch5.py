# =========
# Exc 5.7.1
# =========

import numpy as np

x_vals = np.random.standard_normal(20)
y_vals = np.random.standard_normal(20)

cross_prod = 0
for x, y in zip(x_vals, y_vals):
    cross_prod += x * y
print(cross_prod)

x2 = range(0, 10)
x_even = sum(x % 2 == 0 for x in x2)
print(x_even)

pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
count = 0
for x, y in pairs:
    count += 1 if x % 2 == 0 and y % 2 == 0 else 0
print(count)

print(sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs]))

# =========
# Exc 5.7.2
# =========


def eval_poly(x, an):
    res = 0
    for ind, a in enumerate(an):
        res += a * x**ind
    return res


print(eval_poly(1, (2, 4)))

# =========
# Exc 5.7.3
# =========

test_str = "The Rain in Spain"

print(sum([x == x.upper() and x.isalpha() for x in test_str]))


# =========
# Exc 5.7.4
# =========


def cmp_str(seq_a, seq_b):
    for x in seq_a:
        if x not in seq_b:
            return False
    return True


a = "abc"
b = "abcd"
c = "bcde"

print(cmp_str(a, b))
print(cmp_str(a, c))


# =========
# Exc 5.7.5
# =========


def lin_interpolate(f, a, b, x, n):
    len_interval = b - a
    num_sub_inter = n - 1
    step = len_interval / num_sub_inter

    point = a
    while point <= x:
        point += step

    u, v = point - step, point

    return f(u) + (x - u) * (f(v) - f(u)) / (v - u)


print(lin_interpolate(lambda x: x**2, 1, 5, 2.3, 100))


# =========
# Exc 5.7.6
# =========
n = 100
e_vals = [np.random.standard_normal() for i in range(n)]
