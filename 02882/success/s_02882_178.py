from math import *

EPS = 1e-9

a, b, x = map(int, input().split())
x = x / a
half = a * b / 2
ans = 0

if fabs(x - half) < EPS:
    ans = 45
elif x - half < EPS:
    c = 2 * x / b
    ans = degrees(atan(b / c))
elif x - half > EPS:
    c = 2 * x / a - b
    ans = degrees(atan((b - c) / a))

print(ans)
