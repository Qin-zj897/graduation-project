def mod_inv(n: int, mod: int)->int:
    b, u, v = mod, 1, 0
    while b > 0:
        t = n // b

        n -= t * b
        u -= t * v

        n, b = b, n
        u, v = v, u

    return (u+mod) % mod


def cell_distance(N: int, M: int, K: int)->int:
    MOD = 10**9 + 7

    C = 1
    for k in range(K-2):
        C *= (N*M - K + k + 1)
        C %= MOD
        C *= mod_inv(k+1, MOD)
        C %= MOD

    S = 0
    for d in range(N):
        S += d * (N-d) * M * M
        S %= MOD

    for d in range(M):
        S += d * (M-d) * N * N
        S %= MOD

    S *= C
    S %= MOD

    return S


if __name__ == "__main__":
    N, M, K = map(int, input().split())

    ans = cell_distance(N, M, K)
    print(ans)
