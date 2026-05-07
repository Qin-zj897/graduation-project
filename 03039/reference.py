# 固定为较高效的 success 样本版本：预处理阶乘与逆阶乘，
# 时间复杂度约 O(N*M + N + M)，空间复杂度约 O(N*M)。

# N, M から相異なる二点を取った時のマンハッタン距離の、二点の取り方についての総和を、
# _{N + M - 2} C _{K - 2} 倍したものが答え。

N, M, K = map(int, input().split())

MOD = 10 ** 9 + 7
table_len = 2 * 10 ** 5 + 10

fac = [1, 1]
for i in range(2, table_len):
    fac.append(fac[-1] * i % MOD)

finv = [0] * table_len
finv[-1] = pow(fac[-1], MOD - 2, MOD)
for i in range(table_len - 1, 0, -1):
    finv[i - 1] = finv[i] * i % MOD

distsum1 = 0
for l in range(1, N):
    distsum1 += l * (N - l)
    distsum1 %= MOD
distsum1 *= pow(M, 2, MOD)
distsum1 %= MOD

distsum2 = 0
for l in range(1, M):
    distsum2 += l * (M - l)
    distsum2 %= MOD
distsum2 *= pow(N, 2, MOD)
distsum2 %= MOD

print((distsum1 + distsum2) % MOD * fac[N * M - 2] % MOD * finv[K - 2] % MOD * finv[N * M - K] % MOD)
