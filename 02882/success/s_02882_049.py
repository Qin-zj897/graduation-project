import math 
a, b, x=map(int, input().split())
if x<a*a*b/2:
    angle=180*math.atan(a*b*b/2/x)/math.pi
    print('{:.10f}'.format(angle))
if x>=a*a*b/2:
    angle=180*math.atan((b-(x/a**2))/a*2)/math.pi
    print('{:.10f}'.format(angle))