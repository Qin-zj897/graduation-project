import math
a,b,x=map(int,input().split())
c=float(x/a**2)
d=b-c
e=math.sqrt(a**2+(2*d)**2)
ans=math.degrees(math.acos(2*d/e))
print(90-ans)