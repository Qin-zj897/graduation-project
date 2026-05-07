import math
a,b,x = map(int,input().split())
s1 = a * b**2 / (x+x)
s2 = (2*(a*a*b - x)) / (a**3)
s1 = math.degrees(math.atan(s1))
s2 = math.degrees(math.atan(s2))
if x+x < a*a*b:
    print(s1)
else:
    print(s2)