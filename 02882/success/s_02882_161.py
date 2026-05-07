import math
a,b,x=map(float, input().split())

n=x/a/a
if n>b/2:
    n=(b-n)*2
    atan = math.degrees(math.atan(n/a))
    print('{:.10f}'.format(atan))
else:
    n=n/b*2*a
    atan = math.degrees(math.atan(b/n))
    print('{:.10f}'.format(atan))