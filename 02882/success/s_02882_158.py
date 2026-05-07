import math
a,b,x=map(int,input().split())

if x<=a*a*b/2:
    c=x*2/(a*b)
    theta=math.atan(c/b)
    theta*=360/(2*math.pi)
    theta=90-theta
else:
    c=(b-x/(a*a))*2
    theta=math.atan(c/a)
    theta*=360/(2*math.pi)
print(theta)
