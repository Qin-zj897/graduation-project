N,M,K=map(int,input().split())
mod=10**9+7
X=N*M%mod
C=[1]
for i in range(K+1):
  C.append(C[i]*(X-2-i)*pow(i+1,mod-2,mod)%mod)
P=0
for i in range(N):
  P=(P+(i*(i+1)//2)+((N-i)*(N-i-1)//2))%mod
P=P*M*M%mod
Q=0
for i in range(M):
  Q=(Q+(i*(i+1)//2)+((M-i)*(M-i-1)//2))%mod
Q=Q*N*N%mod
print((P+Q)*C[K-2]*((mod+1)//2)%mod)