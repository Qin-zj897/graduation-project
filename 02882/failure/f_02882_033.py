import math

a, b, x = map(int, input().split(' '))

x = x / a

if 2 * x < a * b:
    print(math.atan(b ** 2 / 2 * x) * 180 / math.pi)
else:
    print(math.atan(2 * (a * b - x) / a ** 2) * 180 / math.pi)
