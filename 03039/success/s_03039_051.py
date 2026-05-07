mod = 10**9+7
inv_2 = pow(2, mod-2, mod)
N, M, K = map(int, input().split())
def make_fact(n):#0~nの階乗を求める
    fact = [1]*(n+1)
    for i in range(1, n+1):
        fact[i] = fact[i-1]*i%mod
    return fact

def make_fact_inv(n):#0~nの階乗のmodに関する逆元を求める
    fact_inv = [1]*(n+1)
    fact_inv[n] = pow(fact[n], mod-2, mod)#フェルマーの小定理
    for i in range(n, 0, -1):
        fact_inv[i-1] = fact_inv[i]*i%mod
    return fact_inv

def comb(n, k):#nCk
    return fact[n]*fact_inv[k]*fact_inv[n-k]%mod
fact = make_fact(N*M)
fact_inv = make_fact_inv(N*M)
Sum = [0]*(max(N, M)+1)
for i in range(1, len(Sum)):
    Sum[i] = (Sum[i-1]+i)%mod
ans = 0

for x in range(1, N+1):
    for y in range(1, M+1):
        ans+=(Sum[x-1]+Sum[N-x])*M+(Sum[y-1]+Sum[M-y])*N
        ans%=mod
print(ans*comb(N*M-2, K-2)*inv_2%mod)