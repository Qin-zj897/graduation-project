N, M, K = map(int,input().split())
MOD = 10**9+7

def cmb(n, r):
    if n - r < r: r = n - r
    if r == 0: return 1
    if r == 1: return n

    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]

    for p in range(2,r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1,r,p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot

    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])

    return result

# 0~N*M-1までの1次元上と考える
ans = 0

def f(x):
    return x*(x+1)//2

c = cmb(N*M-2,K-2)

for i in range(N*M):
    tate, yoko = i//M, i%M
    migi = f(M-1-yoko)
    hidari = f(yoko)
    sita = (hidari+migi)*(N-1-tate) + M*f(N-1-tate)
    coef = migi + sita
    ans += coef*c
    ans %= MOD
print(ans)


    
    
    