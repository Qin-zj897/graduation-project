N=10**6
p=10**9+7
f,finv,inv=[0]*(N*5),[0]*(N*5),[0]*(N*5)

def nCr_init(L,p):
    for i in range(L+1):
        if i<=1:
            f[i],finv[i],inv[i]=1,1,1
        else:
            f[i]=(f[i-1]*i)%p
            inv[i]=(p-inv[p%i]*(p//i))%p
            finv[i]=(finv[i-1]*inv[i])%p

def nPr(n,r,p):
    if 0<=r<=n and 0<=n:
        return (f[n]*finv[n-r])%p
    else:
        return 0
    
nCr_init(N,p)

N,M,K=map(int,input().split())

print((nPr(N*M,K,p)*((M+N)*inv[6]*finv[K-2])%p)%p)