import math
N, M, K = map(int, input().split())
sosu = 10 ** 9 + 7

def P(n, r):
    return math.factorial(n)//math.factorial(n-r)

def C(n, r):
    return P(n, r)//math.factorial(r)


dis = M * M * N * N * (N+1) // 2 % sosu
dis = dis - M * M * N * (N+1) * ( 2* N + 1) // 6 % sosu
dis = dis + N * N * M * M * (M+1) //2 % sosu
dis = dis - N * N * M * (M+1) * (2*M + 1) // 6 % sosu

def cmb(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

mod = sosu #出力の制限
N = N * M
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, N + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )


if K >= 3:
    dis = dis * cmb(N-2, K-2, mod) % sosu
if dis >= 0:
    print(dis)
else:
    print(dis + sosu)