from math import atan2,degrees
a,b,x=map(int,input().split())
x/=a
if 2*x>a*b:
  print(degrees(atan2((a*b-x)*2,a*a)))
else:
  print(degrees(atan2(b*b,x*2)))
