import math
t=input().split(' ')
a=int(t[0]);b=int(t[1]);x=int(t[2])
if x>=a**2*b*0.5:
    print(math.degrees(math.atan(2*b/a-2*x/a**3)))
else:
    print(math.degrees(math.atan(a*b**2/(2*x))))