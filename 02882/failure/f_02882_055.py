# D - Water Bottle
import math
a,b,x = map(int, input().split())

tmp = x/(a**2)
# tmp2 = x/(b**2)
tmp2 = ((2*x)/b)**(1/2)

if tmp2 < a:
    print(90-math.degrees(math.atan2(tmp2,b)))
else:
    #print(tmp)
    tmp3 = (2*b-((2*x)/(a*a)))
    #print(tmp3)
    print(90-math.degrees(math.atan2(a,tmp3)))