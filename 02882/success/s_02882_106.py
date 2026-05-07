import math

a, b, x = map(int, input().split())

PI = math.acos(-1)
ans = 0

def radianToDegree(p):
    return math.degrees(p)

if (a*a*b)/2 <= x:
    ans = math.atan(2 * (a * a * b - x) / (a * a * a))
else:
    ans = PI/2 - math.atan((2*x) / (a*b*b))

print(radianToDegree(ans))
