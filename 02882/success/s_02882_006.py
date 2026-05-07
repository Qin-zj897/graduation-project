from math import atan, pi
a, b, x = map(int, input().split())

if x > (a ** 2 * b / 2):
    print(atan(2 * (a ** 2 * b - x) / a ** 3) * 180 / pi)
else:
    print(atan((a * b ** 2) / (2 * x)) * 180 / pi)