from fractions import math
a, b, x = map(int, input().split())
print(math.atan(a*(b**2)/2/x)/math.pi*180)
