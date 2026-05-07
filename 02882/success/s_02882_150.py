import math

a,b,x = map(int,input().split())	

Z=a*a*b
if x>(Z/2):
    theta=math.atan(((2*Z)-(2*x))/(a*a*a))
    
else:
    theta=math.atan((2*x)/(a*b*b))
    theta=1/2*(math.pi)-theta


print(theta*180/math.pi)