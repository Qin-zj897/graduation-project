from math import atan,pi
a,b,x = map(int,input().split())

ans = 0
if x >= a*a*b/2:
    ans = atan(2*b/a-2*x/a**3)/pi*180
else:
    ans = atan(a*b**2/2/x)/pi*180

print(ans)