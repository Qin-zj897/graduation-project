import math
a,b,x=map(int,input().split())

if (a*a*b/2<=x):
  if (a*a*b==x):
    print(0)
  else:
    theta=math.atan((a*a*a)/(2*(a*a*b-x)))
    print(90.0-theta*180.0/math.pi)
else:
  theta=math.atan(2*x/(a*b*b))
  print((math.pi/2-theta)*180.0/math.pi)