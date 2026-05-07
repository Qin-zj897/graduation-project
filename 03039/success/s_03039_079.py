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

N, M, K = map(int, input().split())

ans = 0

for i in range(1, M):
    ans += (i * (M-i) * N**2) % (10**9 + 7) 
for i in range(1, N):
    ans += (i * (N-i) * M**2) % (10**9 + 7)

ans *= comb(N*M-2 , K-2, 10**9+7)

print(int(ans % (10**9+7)))

