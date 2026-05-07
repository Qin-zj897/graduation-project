import numpy as np

a,b,x = map(int,input().split())
if a*a*b > x*2:
    print(90-np.rad2deg(np.arctan(2*x/(a*b*b))))
else:
    print(np.rad2deg(np.arctan(2*(a*a*b-x)/(a**3))))