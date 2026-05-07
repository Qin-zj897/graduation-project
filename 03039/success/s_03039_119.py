N,M,K=map(int, input().split())
div=10**9+7

def modinv(a, m):
    b = m
    u = 1
    v = 0
    while (b):
        t = a // b
        a -= t * b 
        a = a^b
        b = a^b
        a = a^b
        u -= t * v
        u = u^v
        v = u^v
        u = u^v
    u %= m 
    if (u < 0): u += m
    return u

def ncr(n, r):
    res = 1
    for i in range(1, r+1):
        res = res*n*modinv(i, div) % div
        n = n-1
    return res % div

result = 0
comb = ncr(N*M-2, K-2)
for d in range(1, N):
    result += d*comb*M*M*(N-d)

for d in range(1, M):
    result += d*comb*(M-d)*N*N

print(result % div)
