def cmb(n,r):
    if n-r<r:
        r=n-r
    if r==0:
        return 1
    if r==1:
        return n
    
    numerator=[n-r+k+1 for k in range(r)]
    denominator=[k+1 for k in range(r)]
    
    for p in range(2,r+1):
        pivot=denominator[p-1]
        if pivot>1:
            offset=(n-r)%p
            for k in range(p-1,r,p):
                numerator[k-offset]/=pivot
                denominator[k]/=pivot
    result=1
    for k in range(r):
        if numerator[k]>1:
            result*=int(numerator[k])
    return result
mod=10**9+7
N,M,K=map(int,input().split())
retu=0
gyou=0
for i in range(1,M):
    retu+=i*(M-i)*N*N
for j in range(1,N):
    gyou+=j*(N-j)*M*M
cost=retu+gyou
num=cmb(N*M-2,K-2)%mod
ans=int((num*(cost %mod))%mod)
print(ans)