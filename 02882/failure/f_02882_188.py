import sys
import os
import math

a,b,x = map(int,input().split())
if x >= (a**2)*b/2:
    ｙ = 2*(a*a*b-x)/(a*a)
    print((math.degrees(math.atan2(y,a))))
else:
    print(1)
    y = 2*x/(a*b)
    print((math.degrees(math.atan2(b,y))))
