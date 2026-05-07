from math import atan,pi

a,b,x = map(int,input().split())

if 2*x/a <= a*b:
    theta = atan(a*b**2/(2*x))*180/pi
else:
    theta = atan(2*(b/a-x/a**3))*180/pi

print(theta)
