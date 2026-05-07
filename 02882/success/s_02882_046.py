from math import atan,degrees
a,b,x = map(int,input().split())
s = a**2
if x>=s*b/2:
  k = 2*x/s-b
  print(degrees(atan((b-k)/a)))
else:
  i = 2*x/a/b
  print(degrees(atan(b/i)))