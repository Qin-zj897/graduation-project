def factorials(M):#M!までのリストｘと其の逆元のリストyを返すmod INF
    INF=int(1e9+7)
    fac=[0]*(M+1)
    finv=[0]*(M+1)
    inv=[0]*(M+1)
    fac[0]=fac[1]=1
    finv[0]=finv[1]=1
    inv[1]=1
    for i in range(2,M+1):
        fac[i]=(fac[i-1]*i)%INF
        inv[i]=INF-(inv[INF%i]*(INF//i))%INF
        finv[i]=finv[i-1]*inv[i]%INF
    return fac,finv
def nCr(n,r):
    INF=int(1e9+7)
    if n<r or r<0 or n<0 :
        return 0
    return fac[n]*(finv[r]*finv[n-r]%INF)%INF

N,M,K=map(int,input().split())
MOD=10**9+7
fac,finv=factorials(N*M)
ans=0
for d in range(1,N):
    ans+=d*nCr(N*M-2,K-2)*(((N-d)*M**2)%MOD)
    ans%=MOD
for d in range(1,M):
    ans+=d*nCr(N*M-2,K-2)*(((M-d)*N**2)%MOD)
print(ans%MOD)
