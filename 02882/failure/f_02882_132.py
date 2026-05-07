import math

a, b, x = map(int, input().split())

t = 2*(b*a*a - x)/a/a/a
theta = math.atan(t) * 180 / math.pi

print(theta)