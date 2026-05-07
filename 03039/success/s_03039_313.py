import math
def com(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
N,M,K = map(int,input().split())
mod = 10**9+7

ans=0
for i in range(M):
  l=(0+i)*(i+1)+(M-1-i)*(M-i)
  ans+=l//2*N*N%mod
  
for i in range(N):
  l=(0+i)*(i+1)+(N-1-i)*(N-i)
  ans+=l//2*M*M%mod

         
print(ans*com(N*M-2,K-2)//2%mod)