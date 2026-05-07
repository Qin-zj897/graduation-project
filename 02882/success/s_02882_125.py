import math

a,b,x = (int(a) for a in input().split())
if 2 * x <= a**2 * b :
    t = b
    s = 2 * x / (a*b)
    r = s / t
    print(90 - math.degrees(math.atan(r)))
else : 
    t = a
    s = 2 * b - 2 * x / (a ** 2)
    r = s / t
    print(math.degrees(math.atan(r)))

