MOD=10**9+7

def com(n,k,mod):
    res=1
    tmp=1
    for i in range(1,k+1):
        res=res*(n-i+1)%mod
        tmp=tmp*i%mod
    a=pow(tmp,mod-2,mod)
    return res*a%mod

N,M,K=map(int,input().split())
C=com(N*M-2,K-2,MOD)

X,Y=0,0
for d in range(1,N):
    X=(X+((N-d)*M**2*d))%MOD
for d in range(1,M):
    Y=(Y+((M-d)*N**2*d))%MOD
print((X+Y)*C%MOD)