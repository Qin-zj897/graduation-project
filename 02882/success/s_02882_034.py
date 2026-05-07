import math

a, b, x = map(int, input().split())
if(x <= a*a*b*0.5):
    t = math.atan(0.5*a*b*b/x)
else:
    t = math.atan(2*(a*a*b-x)/(a*a*a))
print(math.degrees(t))