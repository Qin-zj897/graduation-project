#E問題
N,M,K = map(int,input().split())
mod = 1000000007

zen = 1
for i in range(N*M-2):
    zen*=(i+1)
kk = 1
for i in range(K-2):
    kk*=(i+1)
zk = 1
for i in range(N*M-K):
    zk*=(i+1)
    
C = zen//(kk*zk)


X = 0
for i in range(N-1):
    d = i+1
    X+=(N-d)*M*M*d
    X%=mod
Y = 0
for i in range(M-1):
    d = i+1
    Y+=(M-d)*N*N*d
    Y%=mod
print(C*(X+Y)%mod)