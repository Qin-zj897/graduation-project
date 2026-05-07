from math import atan, pi
a, b, x = map(int, input().split())
h = x / (a*a)

l = 2*h - b
c = 2 * a * h / b

if l >= 0:
    print("%.10f"%(180 / pi * atan((b-l)/a)))
else:
    print("%.10f"%(180 / pi * atan(b/c)))