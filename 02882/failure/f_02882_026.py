import math
a,b,x=map(int,input().split())
if a**2*b>=2*x:
  ans=math.degrees(math.atan(2*x/(a*b**2)))
  print(ans)
else:
  ans=math.degrees(math.atan(a**2/(2*a*b-2*x/a)))
  print(ans)
  

