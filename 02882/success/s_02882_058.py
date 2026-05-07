import math
a,b,x=map(int,input().split())
if x>=a*a*b/2:
  c=(-2*x/(a*a)+2*b)/a
else:
  c=(a*b*b)/(2*x)
#print(c)
asin=math.degrees(math.atan(c))
print(asin)