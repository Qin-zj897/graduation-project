import math

a,b,x = map(int, input().split())

V = (a**2)*b

if x < V/2:
    t = (2*x)/(a*b**2)
    print(90 - math.degrees(math.atan(t)))
else:
    t = (2*b/a) - (2*x/a**3)
    print(math.degrees(math.atan(t)))