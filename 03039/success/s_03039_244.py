mod=10**9+7
MAX=2*10**5+100

g1=[1,1]
g2=[1,1]
for i in range(2,MAX+1):
    num_1=g1[-1]*i%mod
    g1.append(num_1)
    g2.append(pow(num_1,mod-2,mod))
    
def cmb(n,r):
    return g1[n]*g2[r]*g2[n-r]%mod


N,M,K=map(int,input().split())

ans=0
for t in range(1,M):
    ans=(ans+t*(N**2)*(M-t))%mod
for t in range(1,N):
    ans=(ans+t*(M**2)*(N-t))%mod

print(ans*cmb(N*M-2,K-2)%mod)