# N, M から相異なる二点を取った時のマンハッタン距離の、二点の取り方についての総和を、
# _{N + M - 2} C _{K - 2} 倍したものが答え。


N, M, K = map(int, input().split())

# コンビネーション前計算。
MOD = 10 ** 9 + 7
table_len = 2 * 10 ** 5 + 10

fac = [1, 1]
for i in range(2, table_len):
    fac.append(fac[-1] * i % MOD)

finv = [0] * table_len
finv[-1] = pow(fac[-1], MOD - 2, MOD)
for i in range(table_len-1, 0, -1):
    finv[i-1] = finv[i] * i % MOD

# マンハッタン距離の総和は、各距離の部分がカウントされるかを考えればよいので、
# (ex. 左半分と右半分にひとつずつなら、必ず二等分する線を横断する部分がある、みたいな)
# l + r = N に対し lM * rM = M^2 * lr の総和と、
# l + r = M に対し lN * rN = N^2 * lr の総和の和でよい。
# シグマ公式をつかえばO(1)にも出来そうだが、面倒なので省略。

# M^2は最後にまとめる
distsum1 = 0
for l in range(1, N):
    distsum1 += l * (N - l)
    distsum1 %= MOD
distsum1 *= pow(M, 2, MOD)
distsum1 %= MOD

# N^2は最後にまとめる
distsum2 = 0
for l in range(1, M):
    distsum2 += l * (M - l)
    distsum2 %= MOD
distsum2 *= pow(N, 2, MOD)
distsum2 %= MOD

print((distsum1 + distsum2) % MOD * fac[N * M - 2] % MOD * finv[K - 2] % MOD * finv[N * M - K] % MOD)
