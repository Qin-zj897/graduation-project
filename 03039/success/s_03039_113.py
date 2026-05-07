mod = 10**9 + 7
n,m,k = map(int,input().split())
nm = n*m
f = [1]*(nm+1)
for i in range(1,nm+1):
    f[i] = (f[i-1]*i)%mod
def C(n,k):
    x = f[n]
    x *= pow(f[n-k],mod-2,mod)
    x %= mod
    x *= pow(f[k],mod-2,mod)
    x %= mod
    return x
def F(n):
    x = n*n*(n+1)//2
    y = n*(n+1)*(2*n+1)//6
    return (x-y)%mod
res = (m*m%mod*F(n)%mod + n*n%mod*F(m)%mod)%mod
res *= C(n*m-2,k-2)
print(res%mod)