import math
a,b,x = map(int,input().split())
if a**2*b>x> (a**2)*b/2:
  tan=2*((a**2)*b-x)/(a**3)
elif x<=a*a*b/2:
  tan=a*b*b/(2*x)
elif x==a*a*b:
  tan=0
  

print(math.degrees(math.atan(tan)))