def comb(n, k):
    x, y = 1, 1
    if 2*k > n:
        k = n - k
    for i in range(k):
        x *= n - i
        y *= i + 1
    return x//y

N, M, K = map(int, input().split())
print((sum(i*(N-i)*M**2 for i in range(N))+sum(i*(M-i)*N**2 for i in range(M)))*comb(N*M-2, K-2) % (10**9+7))