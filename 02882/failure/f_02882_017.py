from math import pi, tan
EPS = 0.0000000001

def f(a, b, theta):
    if theta >= pi * 0.5:
        return 0.0
    elif tan(theta) >= b / a:
        return a * b**2 * 0.5 / tan(theta)
    else:
        return a**2 * b - a**2 *b * 0.5 * tan(theta)

a, b, x = map(int, input().split())

l = 0
r = pi * 0.5
while r - l >= EPS:
    mid = (l + r) * 0.5
    if f(a, b, theta) >= x:
        l = mid
    else:
        r = mid

print("{:.10f}".format(l / pi * 180))
