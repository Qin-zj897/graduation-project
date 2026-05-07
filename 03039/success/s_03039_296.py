mdl = 10**9 + 7

N, M, K = map(int,input().split())

nm = N*M
mm = M*M % mdl
nn = N*N % mdl

U = 2*10**5
fact = [1] * (U+1)

for i in range(1,U+1):
    fact[i] = (fact[i-1]*i) % mdl

def comb(n,k):
    x = fact[n]
    x *= pow(fact[n-k], mdl-2, mdl)
    x %= mdl
    x *= pow(fact[k], mdl-2, mdl)
    x %= mdl
    return x

def F(N):
    x = N*(N+1) // 2 * N
    y = N*(N+1)*(2*N+1) // 6
    return (x-y) % mdl

ans = mm*F(N) % mdl + nn*F(M) % mdl
ans %= mdl
ans *= comb(nm-2,K-2)
ans %= mdl

print(ans)