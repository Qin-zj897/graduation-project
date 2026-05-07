from scipy.special import comb
N, M, K = map(int, input().split())

ans = 0

for i in range(1, M):
    ans += i * (M-i) * N**2 
for i in range(1, N):
    ans += i * (N-i) * M**2

ans *= comb(N*M-2, K-2, exact=True)

print(int(ans % (10**9+7)))
