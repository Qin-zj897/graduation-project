N,M,K=map(int,input().split())
L=N*M+10
mod=10**9+7
table=[1]*(L+3)
t=1
for i in range(1,L+3):
    t*=i
    t%=mod
    table[i]=t
rtable=[1]*(L+3)
t=1
for i in range(1,L+3):
    t*=pow(i,mod-2,mod)
    t%=mod
    rtable[i]=t
ans=0
t=table[N*M-2]*rtable[K-2]*rtable[N*M-K]
for i in range(1,M):
    ans+=N*N*(M-i)*i
    ans%=mod
for i in range(1,N):
    ans+=M*M*(N-i)*i
    ans%=mod
print((ans*t)%mod)