import sys
sys.setrecursionlimit(10**8)

MOD = 10**9 + 7
def power_mod(a,b,p):
    if b==0:
        return 1
    if b%2==0:
        d=power_mod(a,b//2,p)
        return (d*d)%p
    else:
        return(a*power_mod(a,b-1,p)%p)

def inverse(a,p):
    return power_mod(a,p-2,p)

def combination(n, r, p): # nCrを求める
    r = min(n-r, r)
    result = 1
    for i in range(n, n-r, -1):
        result *= i
        result %=p
    for i in range(1, r+1):
        result *= inverse(i,p)
        result %= p
    return result

n,m,k = map(int,input().split())

ans = 0
for i in range(n):
    ans +=(n-i)*(m**2)*i
    ans %=MOD
for i in range(m):
    ans +=(m-i)*(n**2)*i
    ans %=MOD

ans *= combination(n*m-2,k-2,MOD)
ans %= MOD

print(ans)
