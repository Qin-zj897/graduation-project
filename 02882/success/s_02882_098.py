import math
a,b,x=map(int,input().split())
if x<a*a*b/2:
    t=a*b*b/2/x
else:
    t=2*b/a-2*x/(a**3)
print(math.atan(t)/math.pi*180)