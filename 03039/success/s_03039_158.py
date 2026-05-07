"""
基本の方針は「各マンハッタン距離が何回足されるか」という考え方で距離の値を積み上げて行けばよい

まず、XとYは独立に考えることにする。
X軸の距離の方から積み上げる。
X1とX2の距離が足される回数は、
A.X1の列とX2の列の座標の組み合わせの数　×　B.その組み合わせが含まれるような座標の組み合わせの数
である。
A　＝　行数**２
B　＝　CMB(N*M-2 , K-2)である

同様の計算をY軸に関しても行えばよい。
"""
def cmb(n, r, p):
    if (r < 0) or (n < r):
        return 0
    r = min(r, n - r)
    return fact[n] * factinv[r] * factinv[n-r] % p

p = 10 ** 9 + 7
N = 10 ** 6  # N は必要分だけ用意する
fact = [1, 1]  # fact[n] = (n! mod p)
factinv = [1, 1]  # factinv[n] = ((n!)^(-1) mod p)
inv = [0, 1]  # factinv 計算用
 
for i in range(2, N + 1):
    fact.append((fact[-1] * i) % p)
    inv.append((-inv[p % i] * (p // i)) % p)
    factinv.append((factinv[-1] * inv[-1]) % p)

N,M,K = map(int,input().split())

ans = 0

#X軸の距離を考える
A = M*M
B = cmb(N*M-2,K-2,p)
for i in range(1,N):
    ans += ((i*i)-(i-1+0)*i//2)*A*B%p
    ans %= p

#Y軸の距離を考える
A = N*N
B = cmb(N*M-2,K-2,p)
for i in range(1,M):
    ans += ((i*i)-(i-1+0)*i//2)*A*B%p
    ans %= p

print(ans)
