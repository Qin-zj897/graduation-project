import math
a,b,x = map(int,input().split())

if a*a*b*0.5 == x:
  print(45)
elif a*a*b*0.5 < x:
  volume = a*a*b - x
  val = 2*volume/(a*a*a)
  print(volume,val)
  print(math.degrees(math.asin(val)))

else:
  volume = x
  val = 2*volume/(a*b*b)
  print(90-math.degrees(math.asin(val)))