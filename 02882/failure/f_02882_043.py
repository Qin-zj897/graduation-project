import math

a, b, x = map(int,input().split())
tan = 2 * x / (a*b*b)
atan1 = 90 - math.degrees(math.atan(tan))

print(atan1)
