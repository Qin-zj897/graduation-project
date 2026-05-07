import math
s = input().split()
a = int(s[0])
b = int(s[1])
x = int(s[2])
if x < a**2*b/2:
    deg = math.degrees(math.atan2(a*b**2, 2*x))
else:
    deg = math.degrees(math.atan2(2*a**2*b-2*x, a**3))
print(deg)