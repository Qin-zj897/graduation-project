import math

a, b, x = map(int, input().split())

if x / (a ** 2 * b) >= 0.5:
    h = b - x / a ** 2
    w = a / 2
else:
    h = b
    w = 2 * x / (a * b)
    
atan1 = math.degrees(math.atan(h / w))
print(atan1)