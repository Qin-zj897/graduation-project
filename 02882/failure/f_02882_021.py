import math
a, b, x =map(int, input().split())
if x > a*a*b/2:
   print(math.degrees(math.tanh(-2/a*(b-x/(a*a)))))
else:
   print(math.degrees(math.tanh(a*b*b/(2*x))))