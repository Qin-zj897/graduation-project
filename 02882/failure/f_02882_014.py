import math


a,b,x=map(int,input().split())

if 2*x<a*a*b:
    sit=math.acos(2*x/(a*b*b))
    print(math.degrees(sit))
elif x==a*a*b:
    print(0)
elif 2*x>a*a*b:
    p1=2*(a*a*b-x)
    p2=a*a*a
    sit=math.atan(p1/p2)
    print(math.degrees(sit))
else:
    print(45.00000)
