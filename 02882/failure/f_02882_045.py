import math
a,b,x = map(int,input().split())
if x>=a*a*b:
    theta = math.degrees(math.atan(2*(x - a*a*b)/a**3))
else:
    theta = math.degrees(math.atan(0.5*a*b*b/x))
print(theta)