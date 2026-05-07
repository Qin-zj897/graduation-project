a, b, x = map(int, input().split())
r = 180 / math.pi
h = x / a ** 2
c = h * 2 - b
d = 2 * x / (a * b)
if h >= b / 2:
    th = math.atan((h - c) / (a / 2)) * r
else:
    th = ((math.pi / 2) - math.atan((d / b))) * r
print(th)