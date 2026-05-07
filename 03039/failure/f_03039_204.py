N, M, K = map(int,input().split())
MOD = 10 ** 9 + 7
ans = 0
C = 1

for i in range(1, K - 1):
    C *= N * M - i - 1
    C //= i
    
for i in range(3, M + N + 1):
    cnt = 0
    a = min(N, i - 1)
    b = i - a
    for _ in range(b, min(i - 1, M) + 1):
        if a == 1 or b == 1:
            cnt += (N - a + 1) * (M - b + 1)
        else:
            cnt += (N - a + 1) * (M - b + 1) * 2
        a -= 1
        b += 1
    ans += (i - 2) * cnt
    
print(ans * C % MOD)
        