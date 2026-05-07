from math import radians, degrees, atan, pi


a, b, x = map(int, input().split())

t = 2 / (a ** 3) * (b * a ** 2 - x)
theta = atan(t)

print(degrees(theta))
