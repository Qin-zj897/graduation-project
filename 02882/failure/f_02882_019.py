from math import atan,degrees
a,b,x=map(int,input().split())
aab2 = a * a * b /2
if x >= aab2:
  print(90-degrees(atan(a/(b-2 *(x-aab2)/(a*a)))))
else:
  print(90-degrees(atan((2*x/(a*b)/b))))