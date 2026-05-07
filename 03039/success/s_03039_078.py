from itertools import combinations

def solve(N, M, K):
    #A = [(n, m) for n in range(N) for m in range(M)]
    r = 0
    for x in range(N):
        r += x * (N - x) * M * M
    for y in range(M):
        r += y * (M - y) * N * N
    return r

def main():
    N, M, K = map(int, input().split())
    g1 = [1, 1]
    g2 = [1, 1]
    inverse = [0, 1]
    mod = 10 ** 9 + 7
    for i in range(2, N * M):
        g1.append((g1[-1] * i) % mod)
        inverse.append((-inverse[mod % i] * (mod // i)) % mod)
        g2.append((g2[-1] * inverse[-1]) % mod)
    t = solve(N, M, 2)
    for k in range(2, K):
        t = t * (N * M - k) * inverse[k - 1] % mod
    print(t)

main()
