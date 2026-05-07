import math

a, b, x = map(int, input().split())

if x >= a*a*b/2:
    l = 2*(b - (x/a**2))
    theta = math.atan(l/a)
else:
    l = (2*x) / (a*b)
    theta = math.atan(b/l)

print(math.degrees(theta))
