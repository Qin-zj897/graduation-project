import math
a, b, x = [int(w) for w in input().split()]

if x > a * a * b / 2:
    rad = math.atan(2*(a ** 2 * b - x) / a ** 3)
else:
    rad = math.atan((a * b ** 2) / (2 * x))

print(math.degrees(rad))
