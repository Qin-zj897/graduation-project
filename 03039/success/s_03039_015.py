N,M,K=map(int,input().split())
cost=0
mod=10**9+7
NM=N*M
kaijo=[0]*(NM-1)
kaijo[0]=1
for i in range(1,NM-1):
    kaijo[i]=(kaijo[i-1]*i)%mod
gyaku=[0]*(NM-1)
gyaku[NM-2]=pow(kaijo[NM-2],mod-2,mod)
for i in range(NM-2,0,-1):
    gyaku[i-1]=(gyaku[i]*i)%mod
def conb(a,b):
    return (kaijo[a]*gyaku[b]*gyaku[a-b])%mod
x=conb(NM-2,K-2)
for d in range(1,N):
    cost=(cost+x*d*(N-d)*(M**2))%mod
for d in range(1,M):
    cost=(cost+x*d*(M-d)*(N**2))%mod
print(cost)