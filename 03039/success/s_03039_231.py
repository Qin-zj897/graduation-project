def solve():
  N, M, K = map(int, input().split())
  mod = 10**9+7
  com = 1
  for i in range(1,K-1):
    com *= N*M-1-i
    com *= pow(i,mod-2,mod)
    com %= mod
  ans = com*M*N*(M*N-1)*(M+N)//6
  return ans%mod
print(solve())