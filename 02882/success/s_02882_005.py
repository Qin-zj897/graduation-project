import math

a,b,x = map(int,input().split())
h = x / (a**2)

if h >= b/2:
    print(math.degrees(math.atan(2*(b-h)/a)))
else:
    print(math.degrees(math.atan(b**2/(2*a*h))))