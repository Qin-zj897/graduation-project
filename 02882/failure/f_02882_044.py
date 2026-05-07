import math

a, b, x = map(int, input().split())

if a ** 2 * b <= x:
    print(math.degrees(math.atan(2 * (a ** 2 * b - x) / (a ** 3))))
else:
    print(180 - 90 - math.degrees(math.atan(2 * x / (b ** 2 * a))))
