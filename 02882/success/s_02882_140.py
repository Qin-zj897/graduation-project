import math
a,b,x=map(int,input().split())
if (a*a*b)>=2*x:
  h=x/(b*(a/2))
  print(90-math.degrees(math.atan(h/b)))
else:
  h=(((x/a)*2)/a)-b
  b-=h
  if b!=0:
    print(90-math.degrees(math.atan(a/b)))
  else:
    print(0)