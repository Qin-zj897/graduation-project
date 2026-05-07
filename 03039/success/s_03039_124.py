N, M, K = map(int, input().split())

mod = 10**9 + 7
table = [1]
for i in range(N*M):
    table.append(table[-1] * (i+1) % mod)


def inv(N):
    return pow(N, mod-2, mod)  # Why mod?


def cmb(n, r):
    return table[n] * inv(table[n-r]*table[r]) % mod


print((M ** 2 * (N ** 3 - N) + N ** 2 * (M ** 3 - M)) // 6 * cmb(M * N - 2, K - 2) % mod)