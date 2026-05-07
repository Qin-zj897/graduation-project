import math

a,b,x = map(int,input().split())

x = x/a

if x > a*b/2:
    ans = math.atan(2*(a*b-x)/a/a)
    ans = math.degrees(ans)

else:
    ans = math.atan(b*b/2/x)
    ans = math.degrees(ans)

print(ans)