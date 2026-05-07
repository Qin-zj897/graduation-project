N,M,K=map(int,input().split())
mod=10**9+7
NM=N*M
kaijo=[0]*(NM+1)
gyaku=[0]*(NM+1)
kaijo[0]=1
for i in range(1,NM+1):
    kaijo[i]=kaijo[i-1]*i%mod
gyaku[NM]=pow(kaijo[NM],mod-2,mod)
for i in range(NM,0,-1):
    gyaku[i-1]=gyaku[i]*i%mod
C=kaijo[NM-2]*gyaku[K-2]*gyaku[NM-K]%mod
ans=0
for k in range(1,M):
    x=k*(M-k)*C*N*N%mod
    ans=(ans+x)%mod
for k in range(1,N):
    x=k*(N-k)*C*M*M%mod
    ans=(ans+x)%mod
print(ans)