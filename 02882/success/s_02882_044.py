import math
a,b,x=map(float,input().split())
if x>a*a*b/2:
  x=a*a*b-x
  h=x*2/a/a
  print(math.degrees(math.atan(h/a)))
else:
  h=x*2/a/b
  print(math.degrees(math.atan(b/h)))