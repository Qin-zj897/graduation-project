import math

tmp = list(map(int,input().split()))
a = tmp[0]
b = tmp[1]
x = tmp[2]

h = x / (a * a)


if h > b / 2:
  t = 2 * h - b
  r = math.degrees(math.atan((b - t) / a))
else:
  y = 2 * x / (a * b)
  r = math.degrees(math.atan(b / y))
  
print (r)