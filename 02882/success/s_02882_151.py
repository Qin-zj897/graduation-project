import math 
a,b,x = map(int,input().split())
 
if x >= a**2*b/2:
    tan = 2*(b/a-x/(a**3))
    y = math.atan(tan)
else :
    tan = a*(b**2)/(2*x)
    y = math.atan(tan)

print(math.degrees(y))