MOD = 10**9 + 7

N,M,K = map(int,input().split())
NM = N*M

U = 2*10**5
fact = [1] * (U+1)
fact_inv = [1] * (U+1)

for i in range(1,U+1):
  fact[i] = (fact[i-1]*i)%MOD

def comb(n,k):
  x = fact[n]
  x *= pow(fact[n-k],MOD-2,MOD)
  x %= MOD
  x *= pow(fact[k],MOD-2,MOD)
  x %= MOD
  return x

def F(N):
  # sum_x,y abs(x-y)
  x = N*(N+1)//2*N
  y = N*(N+1)*(2*N+1)//6
  return (x-y)%MOD

MM = M*M%MOD
NN = N*N%MOD
answer = MM*F(N)%MOD + NN*F(M)%MOD
answer %= MOD
answer *= comb(N*M-2,K-2)
answer %= MOD
print(answer)