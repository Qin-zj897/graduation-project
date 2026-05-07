import math

a, b, x = input().split(" ")
a = int(a)
b = int(b)
x = int(x)

taiseki = a * a * b

if taiseki - x < x:
    takasa = (taiseki - x) * 2 / (a * a)
    # print(takasa)
    print(math.degrees(math.atan(takasa/a)))
else:
    takasa = x * 2 / (a * b)
    # print(takasa)
    print(math.degrees(math.atan(b/takasa)))
