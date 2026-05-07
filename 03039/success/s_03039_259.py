N, M, K = map(int, input().split())
mod = 10**9+7

def cmb(n, r, mod=mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

NN = 10**6 # 使うデータによって変える
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, NN + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )

ans = 0
for n in range(1, N+1):
    le = n*(n-1)//2 + (N-n)*(N-n+1)//2
    le *= M**2
    ans += le

for m in range(1, M+1):
    le = m*(m-1)//2 + (M-m)*(M-m+1)//2
    le *= N**2
    ans += le

ans //= 2
ans %= mod
ans =  ans * cmb(N*M-2, K-2) % mod
print(ans)