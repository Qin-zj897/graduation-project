import math
a,b,x=[int(x) for x in input().rstrip().split()]
S=x/a
if S==a*b:
  print(0)
else:
  if S<=(a*b)/2:
    a1=S/b
    tan=(b**2)/(2*S)
    print(math.degrees(math.atan(tan)))
  else:
    tan=(2*a*b-2*S)/a**2
    print(math.degrees(math.atan(tan)))


