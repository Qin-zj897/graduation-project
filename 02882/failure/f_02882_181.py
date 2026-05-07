import math
PI=3.14159265358979323846
a,b,x=map(int,input().split())
s=x/a
ans=0
a2=0

if s==a*b:
  ans=0
else:
  if s>b*a/2:
    bu=2*s/a-b
    s-=a*bu
    b-=bu
    a2=2*s/b
  else:
    a2=2*s/b
  r=math.atan(a2/b)
  ans=r*(180/PI)
print(ans)
  