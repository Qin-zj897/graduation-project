a, b, x = map(int, input().split())
rect = a**2 * b

import math

if x <= rect / 2:
    s = x / (a**2) * a
    angle = 90 - math.atan((2 * s) / (b**2)) * (180 / math.pi)
else:
    s = (a**2 * b-x)/(a**2) * a
    angle = math.atan((2 * s) / (a**2)) * (180 / math.pi)

print(angle)