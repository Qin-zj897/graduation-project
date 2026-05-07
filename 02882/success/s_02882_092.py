# 2019/10/29
import math

a,b,x=map(int,input().split())
r=(pow(a,2)*b-(2*x))/(a*b)
pi=math.pi

if r>=0:
    ans=math.atan2(b,a-r)
else:
    t=2*(b-(x/pow(a,2)))
    ans=math.atan2(t,a)

print(math.degrees(ans))