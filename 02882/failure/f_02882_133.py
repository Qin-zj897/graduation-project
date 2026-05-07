a,b,x = map(int,input().split())
import math

if (a+b)/2>x:
    print(float(math.degrees(math.atan(float(a*(b**2)/(2*x))))))
else:
    c=2*(b-x/(a**2))
    print(float(math.degrees(math.atan(c/a))))