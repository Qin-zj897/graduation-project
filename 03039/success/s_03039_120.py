mod=10**9+7
n,m,k=map(int,input().split())
s=n*m

factorial=[1]
for i in range(1,s+1):
    factorial.append(factorial[i-1]*i%mod)
inverse=[0]*(s+1)
inverse[-1]=pow(factorial[-1],mod-2,mod)
for i in range(s-1,-1,-1):
    inverse[i]=inverse[i+1]*(i+1)%mod
def comb(n,r):
    if n<r:
        return 0
    return factorial[n]*inverse[r]*inverse[n-r]%mod

c=comb(n*m-2,k-2)
ans=0
for i in range(1,n):
    ans+=i*(n-i)*(m**2)*c
    ans%=mod
for i in range(1,m):
    ans+=i*(m-i)*(n**2)*c
    ans%=mod
print(ans)