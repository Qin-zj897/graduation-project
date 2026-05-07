import numpy as np

a,b,x=map(int,input().split())
S=a**2*b-x

if a**2*b/2 <= x:
    z=S*2/(a**2)
    rad=np.arctan(z/a)
else:
    z=x*2/(a*b)
    rad=np.arctan(b/z)
print(np.rad2deg(rad))
