import math
a,b,x=map(int,input().split())
cap=a**2*b
y=((cap-x)*2)/a**2

if y<=b:
      y=((cap-x)*2)/a**2
      theta=math.degrees(math.atan(y/a))
      print(theta)
else:
      y=(2*x)/(a*b)
      theta=math.degrees(math.atan(y/b))
      print(90-theta)