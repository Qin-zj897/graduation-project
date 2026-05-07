from math import factorial
N, M, K = map(int, input().split())
MOD = 10 ** 9 + 7

sum = 0
for d in range(1, M):
    c = factorial(N*M-2) // factorial(K-2) // factorial(N*M-K)
    sum += (N*N*(M-d) * c * d) % MOD

for d in range(1, N):
    c = factorial(N*M-2) // factorial(K-2) // factorial(N*M-K)
    sum += (M*M*(N-d) * c * d) % MOD

print(sum)
