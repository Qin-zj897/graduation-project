N,M,K = map(int, input().split())
mod = 10**9+7
nm = N*M
n2 = (N**2)%mod
m2 = (M**2)%mod

def cmb(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, 10**6 +1):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )


ans = 0
for x in range(1,M):
    ans += x*(M-x)*n2*cmb(nm-2,K-2,mod)
    ans %= mod

for y in range(1,N):
    ans += y*(N-y)*m2*cmb(nm-2,K-2,mod)
    ans %= mod

print(int(ans))