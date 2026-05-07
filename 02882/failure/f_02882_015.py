import math

a,b,x = map(int,input().split())
if 2*(a**2)*b <= 2*(a**2)*x:
    print(math.degrees(math.atan2(2*((a**2)*b-x), a**3)))
else:
    print(math.degrees(math.atan2(a*(b**2), 2*x)))