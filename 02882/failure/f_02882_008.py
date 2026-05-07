from math import atan, degrees
a, b, x = map(int, input().split())
if pow(a,b)*b == x:
    print(0)
elif b*pow(a,2)/2 <= x:
    print(90-degrees(atan((a/2)/(b-x/pow(a,2)))))
else:
    print(90-degrees(atan((2*x/(a*b))/b)))