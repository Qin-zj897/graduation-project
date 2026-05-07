from math import pi, atan

a, b, x = map(int, input().split())

if a**2 * b * 0.5 <= x:
    theta = atan(2 * (a**2 * b - x) / a**3)
else:
    theta = atan(a * b**2 * 0.5 / x)

print("{:.10f}".format(theta / pi * 180))
