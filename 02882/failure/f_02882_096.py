import math

a,b,x=map(int,input().split())
if b>a or x<a*a*b//2:
    q=math.acos(x/(a*a*b))
else:
    q=math.atan(2*(a*a*b-x)/(a**3))
print('{:.10f}'.format(math.degrees(q)))