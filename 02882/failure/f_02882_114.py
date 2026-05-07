import math
a, b, x = map(int, input().split())

atan0 = math.degrees(math.atan((a*b**2)/(2*x)))

print(atan0)