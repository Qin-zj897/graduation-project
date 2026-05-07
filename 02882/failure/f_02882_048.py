import math
a,b,V=map(int,input().split())
s=V/a
if s>a*b/2:
  t=round((a**3/(2*(a**2*b-V))),5)
else:
  t=round((2*V/(a*b**2)),5)

ans=math.degrees(math.atan(t))
print(90-ans)