mod = 10**9 + 7
 
 
def comb(n, k):
    x, y = 1, 1
    k = min(k, n - k)
    for i in range(k):
        x *= n - i
        x %= mod
        y *= i + 1
        y %= mod
    return x * pow(y, mod - 2, mod) % mod

N, M, K = map(int, input().split())

c = comb(N*M-2,K-2)

ans = c * (M**2 * (N**2 * (N+1)//2 - N*(N+1)*(2*N+1)//6) + N**2 * (M**2 * (M+1)//2 - M*(M+1)*(2*M+1)//6)) % mod

print(ans)