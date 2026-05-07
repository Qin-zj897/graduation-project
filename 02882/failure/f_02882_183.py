import math
a,b,x=map(int,input().split())
theta=0

if x<=a**2*b/2:
	theta=math.degrees(math.atan(2*x/(a*b**2)))
else:
	theta=math.degrees(math.atan(a**3/(2*(a**2*b-x))))

print(90-theta)