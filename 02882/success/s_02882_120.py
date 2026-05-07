import math

a,b,x = map(int, input().split())
box = a*a*b
ans = 0
if x>=box/2:
    z = a*a
    y = 2*x
    y = y/z
    y -= b
    ans = math.degrees(math.atan2(b-y,a))
else:
    z = a*b
    v = 2*x
    v = v/z
    y = a-v
    ans = math.degrees(math.atan2(b, a-y))
print(ans)