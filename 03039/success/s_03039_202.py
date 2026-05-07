class Factorial():
    def __init__(self,n,mod):
        self.mod = mod
        self.factorial = [0 for _ in range(n+1)]
        self.inv = [0 for _ in range(n+1)]
        self.factorial[0] = 1
        self.inv[0] = 1
        for i in range(n):
            self.factorial[i+1] = self.factorial[i]*(i+1)%mod
        self.inv[n] = pow(self.factorial[n],mod-2,mod)
        for i in range(n)[::-1]:
            self.inv[i] = self.inv[i+1]*(i+1)%mod
    def comb(self,m,k): #組み合わせ 0<=k<=m<=n 重複組み合わせはcomb(m+k-1,k)
        return self.factorial[m]*self.inv[k]*self.inv[m-k]%self.mod

from collections import deque

MOD = 1000000007

N,M,K = map(int,input().split())

F = Factorial(N*M,MOD)

print(F.comb(N*M-2,K-2)*(N+M)*N*M*(N*M-1)//6%MOD)