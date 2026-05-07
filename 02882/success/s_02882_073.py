import math
a, b, x = map(float, input().split())


if x <= (a**2 * b) / 2:
    print(math.degrees(math.atan(a * b**2 / (2 * x))))
else:
    print(math.degrees(math.atan(2 * (a**2 * b - x) / a**3)))
