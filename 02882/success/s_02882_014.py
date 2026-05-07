from math import atan, degrees
a,b,x = map(int,input().split())
o = 2 * x / (a*a) - b
if o < 0:
    h = 2 * (x / a) / b
    print(degrees(atan(b/h)))
else:
    print(degrees(atan((b-o)/a)))
