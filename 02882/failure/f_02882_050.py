import math

a,b,x = map(int,input().split())
if x > (a**1 * b)/2:
    print(math.degrees(float(math.atan(2*(a**2 * b-x)/a**3))))
else:
    print(90-math.degrees(float(math.atan(2*x/(a*b**2)))))