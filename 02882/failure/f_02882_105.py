from math import atan, pi

a, b, x = [float(x) for x in input().split()]

v = a * a * b

if v / 2 < x:
    t = atan(2 * (v - x) / (a * a * a))
else:
    # t = atan(v / 2 / x)
    t = pi / 2 - atan(2 * x / v)

print(t * 180.0 / pi)
