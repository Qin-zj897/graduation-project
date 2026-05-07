import math
(a, b, x) = map(int, input().split(' '))
if x >= a*a*b/2:
    print(180 * math.atan(2*b/a-2*x/(a*a*a)) / math.pi)
else:
    print(180 * math.atan(a*b*b/(2*x)) / math.pi)