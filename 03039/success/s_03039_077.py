N, M, K = map(int,input().split())

## コンビネーション計算用 ========
## https://qiita.com/derodero24/items/91b6468e66923a87f39f
mod = int(1e9+7)
g1 = [1, 1] # 元テーブル
g2 = [1, 1] # 逆元テーブル
inverse = [0, 1] # 逆元テーブル計算用テーブル

for i in range( 2, N*M ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )

def combination(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod
## コンビネーション計算用 --------
  

## 二つの駒の距離の総和 = cost_allpairs
var_1 = M**2 * (N-1) * N * (N+1) // 3
var_2 = N**2 * (M-1) * M * (M+1) // 3
cost_allpairs = int((var_1 + var_2) // 2)
cost_allpairs = cost_allpairs % mod ## ここでmod取っておく

## コンビネーションの計算
cmb = combination(N*M-2, K-2, mod)

## 答え
ans = (cost_allpairs * cmb) % mod
print(int(ans))