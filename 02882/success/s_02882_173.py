import math
a,b,x = map(int,input().split())

if x==a*a*b:
  tanθ = 90
elif a*a*b>x>=a*a*b/2:
  tanθ = math.degrees(math.atan(a**2/(2*(a*b-x/a))))
else:
  tanθ = math.degrees(math.atan(2*x/(a*b*b)))

print(90-tanθ)