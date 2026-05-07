from math import atan
from math import degrees
a,b,x=tuple(map(int,input().split()))
v=(a**2)*b
if 2*x>=v:
  h=x/(a**2)
  print(degrees(atan(2*(b-h)/a)))
else:
  print(degrees(atan(a*(b**2)/(x*2))))