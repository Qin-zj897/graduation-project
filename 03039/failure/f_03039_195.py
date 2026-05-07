def cmb(a,b,c):
    b = min(b,a-b)
    num = 1
    for i in range(b):
        num = num*(a-i) % c
    den = 1
    for i in range(b):
        den = den*(i+1) % c
    return num * pow(den,c-2,c) % c
n,m,k = map(int,input().split())
mod = 10**9+7
ans = 0
for i in range(1,n):
    ans += i*(n-i)*m**2*cmb(n*m-2,k-2,mod)
    ans %= mod
for i in range(1,m):
    ans += i*(m-i)*n**2*cmb(n*m-2,k-2,mod)
    ans %= mod
print(ans)
