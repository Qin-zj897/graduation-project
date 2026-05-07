a, b, x = map(int, input().split())

import math

if x/a >= a*b/2:
    h = 2*(a*b - x/a)/a
    print(math.degrees(math.atan(h/a)))
else:
    h = 2*(x/a)/b
    print(math.degrees(math.atan(b/h)))