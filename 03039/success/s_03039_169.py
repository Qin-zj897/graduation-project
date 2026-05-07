class Factorial:
    def __init__(self,n,mod):
        self.f=[1]
        for i in range(1,n+1):
            self.f.append(self.f[-1]*i%mod)
        self.i=[pow(self.f[-1],mod-2,mod)]
        for i in range(1,n+1)[::-1]:
            self.i.append(self.i[-1]*i%mod)
        self.i.reverse()
    def factorial(self,i):
        return self.f[i]
    def ifactorial(self,i):
        return self.i[i]
    def comb(self,n,k):
        return self.f[n]*self.i[n-k]*self.i[k]
n,m,k=map(int,input().split())
M=10**9+7
print((sum(m*m*i*(n-i)for i in range(n))+sum(n*n*i*(m-i)for i in range(m)))*Factorial(n*m,M).comb(n*m-2,k-2)%M)