N, M, K = map(int, input().split())
ans=0
mod = 10**9+7 #出力の制限
max = 2*10**5
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル
for i in range( 2, max + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )
def cmb(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod
for d in range(1,N):
    ans += d*(N-d)*M**2 % mod
for d in range(1,M):
    ans += d*(M-d)*N**2 % mod
ans = ans*cmb(N*M-2, K-2, mod) % mod
print(ans)