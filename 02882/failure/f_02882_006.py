from math import atan, pi


a, b, x = map(int, input().split())

if a * b / 2 * a < x:
    res = (a ** 3) / (2 * ((a ** 2) * b - x))
else:
    res = 2 * x / (a * b * b)

res = atan(1 / res) / pi * 180
print(res)
