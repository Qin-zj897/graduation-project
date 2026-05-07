import math
a,b,x=map(int,input().split())

if x>a*a*b/2:
    h=(2*a*a*b-2*x)/(a*a)
    print(math.atan(h/a)*(180/math.pi))
else:
    h=2*x/(a*b)
    print(math.atan(b/h)*(180/math.pi))