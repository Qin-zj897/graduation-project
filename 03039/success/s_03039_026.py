"""
組み合わせ(nCr)計算
https://qiita.com/derodero24/items/91b6468e66923a87f39f#%E7%95%AA%E5%A4%96%E7%B7%A8
逆元の求め方
http://drken1215.hatenablog.com/entry/2018/06/08/210000
解法@公式放送
https://www.youtube.com/watch?v=SS6kW-d-rJ0&feature=youtu.be&t=7376
"""


def cmb(n, r):
    return (fact[n] * finv_t[r] * finv_t[n - r]) % MOD


MOD = 10 ** 9 + 7

N, M, K = map(int, input().split())

fact = [1, 1]
inv_t = [0, 1]
finv_t = [1, 1]
for i in range(2, N * M - 2 + 1):
    fact.append((fact[-1] * i) % MOD)
    inv_t.append((-inv_t[MOD % i] * (MOD // i)) % MOD)
    finv_t.append((finv_t[-1] * inv_t[-1]) % MOD)

pat = cmb(N * M - 2, K - 2)

ans = 0

for i in range(M):
    ans += i * (M - i) * N ** 2

for i in range(N):
    ans += i * (N - i) * M ** 2

ans = (ans * pat) % MOD

print(ans)
