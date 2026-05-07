from itertools import accumulate
def inpl(): return list(map(int, input().split()))
MOD = 10**9 + 7
N, M, K = inpl()

X = sum([n*(n+1)//2 for n in range(1, N)])*M*(M)%MOD
Y = sum([m*(m+1)//2 for m in range(1, M)])*N*(N)%MOD


def cmb(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod


g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, N*M + 1 ):
    g1.append( ( g1[-1] * i ) % MOD )
    inverse.append( ( -inverse[MOD % i] * (MOD//i) ) % MOD )
    g2.append( (g2[-1] * inverse[-1]) % MOD )

print((X+Y)*cmb(N*M-2, K-2, MOD)%MOD)
