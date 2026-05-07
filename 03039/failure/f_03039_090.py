n,m,k = map(int,input().split())
ans = 0
MOD = 10**9+7
import  numpy as np
x = np.array([i for i in range(1,n+1)]*m)
y = np.array([i for i in range(1,m+1)]*n)
npabs = np.frompyfunc(abs,1,1)
for i in range(len(x)-1):
    x_ = x[i:]-x[i]
    x_ = np.sum(npabs(x_))
    y_ = y[i:]-y[i]
    y_ = np.sum(npabs(y_))
    ans += x_+y_
from scipy.special import comb
print((ans*comb(m*n-2,k-2,exact=True))%MOD)