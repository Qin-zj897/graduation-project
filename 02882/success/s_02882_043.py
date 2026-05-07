import math

a, b, x = map(int, input().split())
res = 0

if x > a**2*b/2:
    res = math.degrees(math.atan(2*(a**2*b-x)/a**3))
else:
    res = math.degrees(math.atan(a*b**2/2/x))

print('{:.10f}'.format(res))