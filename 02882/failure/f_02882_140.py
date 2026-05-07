import math
a,b,x=map(int,input().split())
c=b/2
if x<=a*a*c:
    d=2*x/b/a
    print(round(math.degrees(math.atan(b/d)),10))
if x==a*a*b:
    print(0)
else:
    x=a*a*b-x
    d=2*x/a
    print(round(90-math.degrees(math.atan(a/d)),10))