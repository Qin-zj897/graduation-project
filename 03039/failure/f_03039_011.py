N, M, K = map(int,input().split())
MOD = 10 ** 9 + 7
ans = 0
C = 1

for i in range(1, K - 1):
    C *= N * M - i - 1
    C //= i
    
for i in range(N):
    for j in range(M):
        ad = (N - i) * (M - j) * (i + j)
        if i != 0 and j != 0:
            ad *= 2
            ans += ad    
print(ans * C % MOD)
if __name__ == "__main__":
    main()