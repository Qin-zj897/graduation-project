from scipy.misc import comb

N, M, K = map(int, input().split())

mod = 10 ** 9 + 7
ans = 0
for d in range(1, N):
    ans += d * (N - d) * M**2
for d in range(1, M):
    ans += d * (M - d) * N**2
print(ans * comb(N * M - 2, K - 2, exact=True) % mod)
