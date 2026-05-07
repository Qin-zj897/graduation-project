import math
N,M,K=map(int,input().split())
mod=10**9+7
ans=0
for i in range(1,N):
  l=(N+1-i)*(N-i)//2%mod 
  ans+=l*M*M
  ans%mod
for i in range(1,M):
  l=(M+1-i)*(M-i)//2%mod 
  ans+=l*N*N
  ans%mod
  
def c_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
                                 
ans*=c_count(M*N-2,K-2)%mod
print(ans%mod)