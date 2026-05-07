# ABC144 D
from math import atan2,pi

a,b,x=map(int,input().split())
if a*a*b>2*x:
    h=2*x/(a*b)
    print(90-atan2(2*x,a*b*b)*180/pi)
else:
    print(atan2((2*(a*a*b-x)),a**3)*180/pi)