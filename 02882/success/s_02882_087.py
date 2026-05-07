import numpy as np
a,b,x=map(int, input().split())
V=a*a*b
if(V/2<x):
  print(np.rad2deg(np.arctan(2*(V-x)/(a*a*a))))
else:
  print(90-np.rad2deg(np.arctan(2*x/(a*b*b))))
  