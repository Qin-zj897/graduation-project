
N, M, K = map(int, input().rstrip().split(' '))

comb = [0] * (N * M + 1)
comb[K-2] = 1
for i in range(K-1, N*M+1):
    comb[i] = comb[i-1] * i // (i - (K - 2))
    comb[i] %= 1000000007

cost = 0
for d in range(1, N):
    cost += d * M**2 * (N - d) * comb[N*M-2]# comb[(N-d-1)*M]
    cost %= 1000000007
for d in range(1, M):
    cost += d * N**2 * (M - d) * comb[N*M-2]# comb[(M-d-1)*N]
    cost %= 1000000007
    
print(cost)

