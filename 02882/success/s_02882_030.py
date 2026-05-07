import math

a , b , x = map(int,input().split())

if b >= 2 * (b - (x / (a * a))):
    print(math.degrees(math.atan((2 * (b - (x / (a * a)))) / a)))
else:
    print(math.degrees(math.atan((a * b * b) / (2 * x))))