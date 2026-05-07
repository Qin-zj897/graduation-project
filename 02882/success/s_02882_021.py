import math

a, b, x = map(int, input().split())

if a ** 2 * b == x:
    print(0.0)
elif a ** 2 * b * 0.5 < x:
    d = a ** 3 / (2 * (a ** 2 * b - x))
    print(90 - math.atan(d) * 180 / math.pi)
else:
    d = 2 * x / (a * b ** 2)
    print(90 - math.atan(d) * 180 / math.pi)
