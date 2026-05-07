import math
a,b,x = map(int,input().split())
if 2*x>=a*a*b:
    l = 2*(b-x/a**2)
    print(math.degrees(math.atan(l/a)))
else:
    l = 2*x/a**2
    print(math.degrees(math.atan(a*b**2/2/x)))