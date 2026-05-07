import math
a,b,x=map(int,input().split())
ans=0

if x==a*a*b/2:
  ans=a/b
elif x<a*a*b/2:
  t=a-2*x/(a*b)
  ans=(a-t)/b
else:
  t=2*x/(a**2)-b
  ans=a/(b-t)

print(90-math.degrees(math.atan(ans)))