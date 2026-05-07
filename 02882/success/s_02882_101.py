import math

a, b, x = [int(x) for x in input().split()]

bootle_volume = a * a * b
half_bootle_volume = 0.5 * bootle_volume

res = 0
if half_bootle_volume == x:
    res = math.degrees(math.atan(1))
elif half_bootle_volume > x:
    y = a - ((2 * x) / (a * b))
    degree = math.degrees(math.atan((a - y) / b))
    res = 90 - degree
elif half_bootle_volume < x:
    y = ((2 * x) / (a ** 2)) - b
    if y == b:
        res = 0
    else:
        degree = math.degrees(math.atan(a / (b - y)))
        res = 90 - degree

print(res)
