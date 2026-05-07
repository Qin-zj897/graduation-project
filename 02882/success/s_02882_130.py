import math

a, b, x = (int(_) for _ in input().split())

L3 = b * (a**2)
x2  = L3 - x

if x >= x2:
    h = 2 * (x2 / (a**2))
    print(math.degrees(math.atan(h/a)))
else:
    h = 2 * (x  / (a*b))
    print(90-math.degrees(math.atan(h/b)))
