import math

a,b,x = map(int,input().split())
S = a * b / 2

if S >= x:
  change_length = 2*x/(a**2*b)
  rad = math.atan2(b, change_length)
else:
  change_length = (2*b - 2*x/a**2)
  rad = math.atan2(change_length, a)
Theta = math.degrees(rad)
print(Theta)