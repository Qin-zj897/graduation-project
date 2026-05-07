import numpy as np
A,B,X=map(int,input().split())
X/=A
if A*B/2<=X:
	print(np.rad2deg(np.arctan(2*(A*B-X)/(A*A))))
else:
    print(np.rad2deg(np.arctan(B*B/(2*X))))