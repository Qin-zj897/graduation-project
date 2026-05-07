import math
a,b,x = map(int, input().split())
print(math.degrees(math.atan(a*b*b/2/x)))