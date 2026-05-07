# D - Water Bottle
from math import atan, pi
a, b, x = map(int, input().split())

if x <= a * a * b / 2:
    theta = atan(2 * x / (a * b * b))
else:
    theta = atan((a * a * a) / (2 * (a * a * b - x)))
print(90 - theta * 180 / pi)