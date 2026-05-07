def com(n,r):
    if n-r<r:
        r=n-r
    re=1
    for i in range(n,n-r,-1):
        re*=i
    for i in range(1,r+1):
        re//=i
    return re

md=10**9+7
n,m,k=map(int,input().split())
ans=0
for d in range(1,m):
    ans+=d*(m-d)*n*n
for d in range(1,n):
    ans+=d*(n-d)*m*m
ans=ans*com(n*m-2,k-2)
print(ans%md)
