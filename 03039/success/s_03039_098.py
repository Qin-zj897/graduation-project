N, M, K = [int(_) for _ in input().split()]


def cmb(n, r, mod):
    if (r < 0 or r > n):
        return 0
    r = min(r, n - r)
    return g1[n] * g2[r] * g2[n - r] % mod


mod = 10**9 + 7  #出力の制限
g1 = [1, 1]  # 元テーブル
g2 = [1, 1]  #逆元テーブル
inverse = [0, 1]  #逆元テーブル計算用テーブル

for i in range(2, N * M + 1):
    g1.append((g1[-1] * i) % mod)
    inverse.append((-inverse[mod % i] * (mod // i)) % mod)
    g2.append((g2[-1] * inverse[-1]) % mod)


def count(n, m):
    ret = 0
    for d in range(1, n):
        ret += d * (n - d) * m**2
    return ret


ans = cmb(N * M - 2, K - 2, mod)
ans *= count(N, M) + count(M, N)
ans %= mod
print(ans)
