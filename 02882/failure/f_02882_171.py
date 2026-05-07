import math
a, b, x = map(int, input().split())
v = a * a * b
if 2 * x <= v:
    print('a')
    print(math.degrees(math.atan((a*b*b) / (2*x))))
else:
    print(math.degrees(math.atan((2 * (v - x) / (a * a)) / a)))

