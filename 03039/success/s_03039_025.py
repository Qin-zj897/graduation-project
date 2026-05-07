N, M, K = map(int,input().split())
MOD = 10**9+7

def framod(n, mod, a=1):
    for i in range(1,n+1):
        a=a * i % mod
    return a

def power(n, r, mod):
    if r == 0: return 1
    if r%2 == 0:
        return power(n*n % mod, r//2, mod) % mod
    if r%2 == 1:
        return n * power(n, r-1, mod) % mod

def comb(n, k, mod):
    a=framod(n, mod)
    b=framod(k, mod)
    c=framod(n-k, mod)
    return (a * power(b, mod-2, mod) * power(c, mod-2, mod)) % mod

# 0~N*M-1までの1次元上と考える
ans = 0

def f(x):
    return x*(x+1)//2

c = comb(N*M-2,K-2,MOD)

for i in range(N*M):
    tate, yoko = i//M, i%M
    migi = f(M-1-yoko)
    hidari = f(yoko)
    sita = (hidari+migi)*(N-1-tate) + M*f(N-1-tate)
    coef = migi + sita
    ans += coef*c
    ans %= MOD
print(ans)


    
    
    