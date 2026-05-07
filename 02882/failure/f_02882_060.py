import math
a, b, x = map(int, input().split())
n = a * b
x /= a
if x / n < 0.5:
    c = (2 * (n - x)) / b - a
    d = a - c
    atan = math.degrees(math.atan(d / b))
else:
    c = 2 * x / a - b
    d = b - c
    atan = math.degrees(math.atan(a / d))

print(90 - atan)