n,m,k=map(int,input().split())
mod=10**9+7
A=n*m-2
from math import factorial
def Comb(N, K):
    a = factorial(N) % mod
    b = factorial(K) % mod
    c = factorial(N-K) % mod
    return (a * pow(b, mod-2, mod) * pow(c, mod-2, mod)) % mod
v=Comb(A,k-2)
t=0
y=0
for i in range(1,n):
  t+=i*(n-i)*m*m%mod
for i in range(1,m):
  y+=i*(m-i)*n*n%mod

ans=v*(t+y)
print(ans%mod)