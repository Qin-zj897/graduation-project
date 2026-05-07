def com(n,r):
    re=fct[n]*pow(fct[r],md-2,md)*pow(fct[n-r],md-2,md)
    re%=md
    return re

n,m,k=map(int,input().split())
md=10**9+7
ans=0
fct=[1]*(n*m-1)
f=1
for i in range(1,n*m-1):
    f=f*i%md
    fct[i]=f
for d in range(1,m):
    ans=(ans+d*(m-d)*n*n)%md
for d in range(1,n):
    ans=(ans+d*(n-d)*m*m)%md
ans=(ans*com(n*m-2,k-2))%md
print(ans)
