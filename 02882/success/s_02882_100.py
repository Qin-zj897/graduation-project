import math
a,b,x=map(int,input().split())
c=b-(x/(a*a))
k=math.sqrt(c**2+(a*a/4))
if x<=a*a*b/2:
  o=(x*2/b)/a
  k=math.sqrt(o**2+b*b)
  print(math.degrees(math.acos(o/k)))
else:
  print(math.degrees(math.acos(a/(2*k))))