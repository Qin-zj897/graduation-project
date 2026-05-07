import math
a, b, x = map(int, input().split())
i = a**3
j = 2*(a*a*b-x)
tan1 = i/j
deg = math.degrees(math.atan(tan1))
if deg < 45:
    tan2 = 2*x / (a*b*b)
    deg = math.degrees(math.atan(tan2))

print(90-deg)