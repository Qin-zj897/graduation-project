from math import atan, pi

eps = 10e-8

a, b, x = [float(x) for x in input().split()]

v = a * a * b

if v / 2 < x - eps:
    t = atan(2 * (v - x) / (a * a * a))
elif v / 2 > x + eps:
    # t = atan(v / 2 / x)
    t = pi / 2 - atan(2 * x / v)
else:
    t = atan(b / a)

print(t * 180.0 / pi)
