def combnk_mod(n, k, p):
    # pが素数かつaとpが互いに素であるとき
    # フェルマーの小定理より以下が成り立つ.
    # 1 / a = a ** (p - 2) (mod p)
    # これを使って (1 / k!) mod p を計算する.

    if  2 * k > n:
        k = n - k
    x = 1 # 分母にあたる数 1 / n! mod p
    y = 1 # 分子にあたる数. n! / (n - k)! mod p
    for i in range(1, k + 1):
        x *= pow(i, p - 2, p)
        y *= n - i + 1
        x %= p
        y %= p
    return (x * y) % p

n, m, k = map(int, input().split())
MOD = 10 ** 9 + 7
p = combnk_mod(n * m - 2, k - 2, MOD)
x, y = 0, 0
for i in range(2):
    for d in range(1, n):
        x += d * (n - d) * m ** 2
        x %= MOD
    x, y = y, x
    n, m = m, n
print(p * (x + y) % MOD)
