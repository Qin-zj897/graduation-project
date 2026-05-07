# D - Water Bottle
import math
a,b,x = map(int, input().split())

tmp = x/(a**2)
# tmp2 = x/(b**2)

#print(tmp)
tmp2 = (2*b-((2*x)/(a*a)))
#print(tmp2)
print(90-math.degrees(math.atan2(a,tmp2)))

