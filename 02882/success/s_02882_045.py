import math
from fractions import Fraction
a,b,x=map(int,input().split())
if b*(a**2)<=x*2:
    print(math.degrees(math.atan(Fraction(2*(a*a*b-x),a**3))))
else:
    print(math.degrees(math.atan(Fraction(a*b*b,2*x))))