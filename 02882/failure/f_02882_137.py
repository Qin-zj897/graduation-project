a, b, x = map(int, input().split())
from math import atan,pi
if x<= a*a*b/2:
    print(90-atan((x*2/b/a)/b)*180/pi)
    exit()
else:
    print(90-atan(a/((a*a*b-x)*2/a/a))*180/pi)
    