import math

a, b, x = map(int, input().split(' '))

x = x / a

if x > a * b / 2:
    print(math.atan2((a * b - x) * 2, a ** 2) * 180 / math.pi)
else:
    print(math.atan2(b ** 2, x * 2) * 180 / math.pi)
