import math
a, b, x = map(int, input().split())
if x == (a**2*b)/2:
    print(45)
elif x > (a**2*b)/2:
    print(math.degrees(math.atan((2*(b-x/(a**2)))/a)))
else:
    print(math.degrees(math.atan(a*b**2/(2*x))))
