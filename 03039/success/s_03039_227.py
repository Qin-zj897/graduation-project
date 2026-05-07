import math
n,m,k=map(int,input().split())
mod=10**9+7
s=math.factorial(n*m-2)//(math.factorial(n*m-k)*math.factorial(k-2))
cnt=0
s%=mod
a=(m*m*s)%mod
for d in range(1,n):
        cnt+=d*(n-d)*a
        cnt%=mod
a=(n*n*s)%mod
for d in range(1,m):
        cnt+=d*(m-d)*a
        cnt%=mod
print(cnt)