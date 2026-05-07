import math

inputs = list(map(int, input().split()))
a = inputs[0]
b = inputs[1]
x = inputs[2]

l = a * a * b
d = x / l
if d <= 0.5:
  h = 2 * a * d
  atan = (math.degrees(math.atan(b/h)))
  print(round(atan, 6))
else:
  h = 2 * b * (1-d)
  atan = (math.degrees(math.atan(h/a)))
  print(round(atan, 6))