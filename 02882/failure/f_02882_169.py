a,b,x = map(int,input().split())
h = x/(a*a)

import math

def f1(d,H,h):
    if h > H/2:
        return math.degrees(math.atan(d/(2*(H-h))))
    else:
        return math.degrees(math.atan((2*d*h)/(H*H)))

print(str.format('{0:.10f}', 90 - f1(a,b,h)))
