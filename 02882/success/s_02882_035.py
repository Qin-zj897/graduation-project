a,b,x = [int(i) for i in input().split()]
from math import atan
from math import degrees
def deg(n):
    return degrees(atan(n))
V = a*a*b
if V/2 == x:
    xa = deg(1)
    xa = str(xa) + '0000000000'
    print(xa)
    exit()
if V/2 > x:
    xa = deg(a*b*b/(2*x))
    print(xa)
    exit()
if V/2 < x:
    xa = deg(2*(V-x)/a**3)
    print(xa)
    exit()