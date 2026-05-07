n, m, k = map(int, input().split( ))

mod = 10**9 + 7

def chs(a,b, md):
    res = 1
    for i in range(a-b+1,a+1):
        res *=  i
        res %= md
    rev = 1
    for i in range(1,b+1):
        rev *= i
        rev %= md
    rev2 = pow(rev,md-2,md)
    
    res *= rev2
    res %= md
    
    return res

ans = 0
c = chs(n*m-2,k-2,mod)

for i in range(1,n):
    ans += i*(n-i)*m**2
    ans %= mod

        

for i in range(1,m):
    ans +=i*(m-i)*n**2
    ans %=mod
ans *= c
ans %= mod
print(ans)
