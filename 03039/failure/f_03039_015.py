from math import factorial
mod=10**9+7
def cmb(n, r):
    return factorial(n)%mod // factorial(r)%mod // factorial(n - r)%mod

n,m,k=map(int,input().split())
print(((pow(m,2,mod)*cmb(n*m-2,k-2)*sum((n-i)*i%mod for i in range(1,n))%mod)+pow(n,2,mod)*cmb(n*m-2,k-2)*sum((m-i)*i%mod for i in range(1,m))%mod)%mod)