import math

a,b,x=map(int,input().split())
if x > a*a*b/2:
    ans=math.atan((2*b/a)-(2*x/a**3))
else:
    ans=math.atan(a*b*b/(2*x))

print(math.degrees(ans))


