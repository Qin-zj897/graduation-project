import math
a,b,x=map(int,input().split())

if a <=b:
    d = (2*x)/(a*b)
    theta = math.atan2(b,d)
    print(math.degrees(theta))
else:
    d = ((2*x)-(b*a*a))/(a*a)
    theta = math.atan2(b-d,a)
    print(math.degrees(theta))
