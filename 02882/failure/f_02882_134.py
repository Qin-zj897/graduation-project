import math 
a, b, x = map(int, input().split())
x = x / a * 2

v = a ** 2 * b

adig = math.degrees(math.atan(
  x / (a ** 2)
))
bdig = math.degrees(math.atan(
  (b ** 2) / x
))
if math.tan(math.radians(adig)) * a <= b:
  print(adig)
else:
  print(bdig)
