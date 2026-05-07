import math
a,b,x = map(int, input().split()) 
if x>=a*a*b/2:
    AA=2*(a*a*b-x)/a**3
    atan = math.degrees(math.atan(AA))
else:
    BB=a*b*b/(2*x)
    atan = math.degrees(math.atan(BB))
print(atan)

