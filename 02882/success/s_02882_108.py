a, b, x = [int(i) for i in input().split()]

import math

h = (2 * x) / (a * b)
if h <= a:
    print(90 - math.degrees(math.atan(h / b)))
else:
    print(math.degrees(math.atan(((((a ** 2) * b) - x) * 2) / (a ** 3))))
