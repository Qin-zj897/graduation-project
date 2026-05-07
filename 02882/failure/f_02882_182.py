import math
a,b,x=map(int,input().split())
if a*b**2>=2*x:
    print(math.degrees(math.atan(a*b**2/(2*x))*180/math.pi))
elif x==0:
    print(90)
else:
    print(math.degrees(math.atan((2*b/a)-(2*x/(a**3)))))