import math

a,b,x=map(int,input().split())

if a*b/2 <= x/a:
  t=(a*b-x/a)*2/(a*a)
  ans=math.degrees(math.atan(t))
  
else:
  t=x*2/(a*b*b)
  ans=90-math.degrees(math.atan(t))
  
  
print(ans)


