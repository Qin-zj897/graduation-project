import math
a,b,x = map(int,input().split())

if x > a*b**2/2:
    y = 2*b - 2*x/(a**2)
    print(math.atan(y/a)*180/math.pi)
else:
    y = 2*x/(a*b)
    print(math.atan(y/b)*180/math.pi)