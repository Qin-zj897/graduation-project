N, M, K = map(int, input().split())

mod = 10**9+7
frac = [1]*200100
for i in range(2,200100):
    frac[i] = i * frac[i-1]%mod
fraci = [None]*200100
fraci[-1] = pow(frac[-1], mod -2, mod)
for i in range(-2, -200100 - 1, -1):
    fraci[i] = fraci[i+1] * (200100 + i + 1) % mod
def comb(a, b):
    if not a >= b >= 0:
        return 0
    return frac[a]*fraci[b]*fraci[a-b]%mod


mod = 10**9+7

t = N**2*(M+1)*M*(M-1)//6 + M**2*(N+1)*N*(N-1)//6
print(t*comb(N*M-2,K-2)%mod)
