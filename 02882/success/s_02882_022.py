import math
a, b, x = map(int, input().split())

if a*a*b == x:
    print(0)
elif a*a*b <= 2*x:
    print(90-math.degrees(math.atan(a*a*a/(2*a*a*b-2*x))))
else:
    print(90-math.degrees(math.atan(2*x/(a*b*b))))
