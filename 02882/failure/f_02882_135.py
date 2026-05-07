import math

a, b, x = map(int, input().split())

y = (a ** 2) * b / 2
if x == a**2*b/2:
    print(0)
elif x <= a**2*b/2:
    z = a*b**2 / (2*x)
    print(math.degrees(math.atan(z)))
else:
    z = 2 * (a ** 2 * b - x) / a ** 3
    print(math.degrees(math.atan(z)))