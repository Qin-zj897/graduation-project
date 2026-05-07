from math import *

a, b, x = list(map(int, input().split()))

th = atan2(a, b)

tmp1 = atan(a**3 / (2 * (a**2 * b - x)))
tmp2 = atan(2 * x / (a * b**2))

if tmp1 <= th:
    print(90 - degrees(tmp2))
else:
    print(90 - degrees(tmp1))
