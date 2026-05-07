n,m,k=map(int,input().split())
ans=0
INF=10**9+7
if n<=m:
    M=-(m-1)*m*(2*m-1)//3+m*m*(m-1)
    Ms=m*m*(m-1)//2-(m-1)*m*(2*m-1)//6
    M=M%INF
    Ms=Ms%INF
    for i in range(1,n+1):
        for j in range(i,n+1):
            if i==j:
                ans=ans+Ms
            else:
                d=j-i
                ans=ans+d*m**2+M
            ans=ans%INF
else:
    (n,m)=(m,n)
    M=-(m-1)*m*(2*m-1)//3+m*m*(m-1)
    Ms=m*m*(m-1)//2-(m-1)*m*(2*m-1)//6
    M=M%INF
    Ms=Ms%INF
    for i in range(1,n+1):
        for j in range(i,n+1):
            if i==j:
                ans=ans+Ms
            else:
                d=j-i
                ans=ans+d*m**2+M
            ans=ans%INF
def modcomb(n,k,m):
    fac=[0]*(n+1)
    finv=[0]*(n+1)
    inv=[0]*(n+1)
    fac[0]=fac[1]=1
    finv[0]=finv[1]=1
    inv[1]=1
    for i in range(2,n+1):
        fac[i]=fac[i-1]*i%m
        inv[i]=m-inv[m%i]*(m//i)%m
        finv[i]=finv[i-1]*inv[i]%m
    return fac[n]*(finv[k]*finv[n-k]%m)%m
ans=ans*modcomb(n*m-2,k-2,INF)%INF
print(ans)

