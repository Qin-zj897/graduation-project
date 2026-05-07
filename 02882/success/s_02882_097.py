import math
a,b,x = map(int,input().split())

s = x / a
height = s / a

if height * 2 > b:
  print(math.degrees(math.atan((b-height)/(a*0.5))))

else:
  tall = s*2/b
  print(math.degrees(math.atan(b/tall)))