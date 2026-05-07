from scipy.special import comb

N, A, B, C = map(int, input().split())
A /= 100
B /= 100
ans = 0

for i in range(100):
    ans += comb(N + i - 1, i) * (N * i) * (A ** N * (1 - A) ** i + B ** N * (1 - B) ** i)
print(ans)