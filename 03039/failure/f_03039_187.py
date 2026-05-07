n,m,k = map(int,input().split())
ans = 0
MOD = 10**9+7
from scipy.misc import comb
C = comb(m*n-2,k-2,exact=True)%MOD
for i in range(1,n):
    ans += (i*(n-i)*(m**2))*C
for i in range(1,m):
    ans += (i*(m-i)*(n**2))*C
print(ans%MOD)