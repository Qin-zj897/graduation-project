import math
a,b,x=(int(x) for x in input().split())

S = x/a
if S >= a*b/2:
    h = 2 * (a * b - S) / a
    print(math.degrees(math.atan2(h, a)))
else:
    w = 2 * S / b
    print(math.degrees(math.atan2(b, w)))