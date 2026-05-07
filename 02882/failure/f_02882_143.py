import math
a,b,x=map(int,input().split())
if x>=2/a**2*b:
  print(math.degrees(math.asin(2*(a**2*b-x)/((2*(a**2*b-x)/a**2)**2+a**2)**0.5)))
else:
  print(math.degrees(math.asin(b/((2*x/a/b)**2+b**2)**0.5)))
