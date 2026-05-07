import math

a,b,x = map(int,input().split())


if a*a*b>2*x:
	print(90- math.atan(2*x/(a*b*b) *360/(2*math.pi)))
else:
	print(90- math.atan(a**3/(2*(a*a*b-x))) *360/(2*math.pi))