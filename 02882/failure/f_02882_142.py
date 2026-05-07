import math
a,b,x = map(int,input().split())
if x >= 1/2 * (a*a*b):
    theta = math.atan(2(a*a*b-x)/(a*a*a))
else:
    theta = 90 - math.atan(2x/(a*b*b))
print(theta)