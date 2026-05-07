import math
a, b, x = map(int, input().split())

if x <= a*a*b/2:
  print(math.degrees(math.asin(a*b*b/((b**4)*(a**2)+4*x*x)**0.5)))
else:
  print(math.degrees(math.asin((2*a*a*b-2*x)/(a**6+4*(a*a*b-x)**2)**0.5)))
        