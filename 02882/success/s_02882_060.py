from math import atan, radians, degrees

a, b, x = map(int, input().split())
h = x/(a**2)
he = b - h

if 2*h >= b:
    ans = degrees(atan((2*b-2*h)/a))
    print(ans)

else:
    ans = degrees(atan(b**2/(2*h*a)))
    print(ans)
