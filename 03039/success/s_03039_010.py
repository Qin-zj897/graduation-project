MOD = 10**9 + 7
MAX = int(2e5)


def div(a, b):
    return a * pow(b, MOD-2, MOD) % MOD


FACT = [1] * (MAX+1)
for i in range(1, MAX+1):
    FACT[i] = (i * FACT[i-1]) % MOD
INV = [1] * (MAX+1)
INV[MAX] = div(1, FACT[MAX])
for i in range(MAX, 0, -1):
    INV[i-1] = (INV[i] * i) % MOD


def combination(r, c):
    return FACT[r] * INV[r-c] * INV[c]


def main():
    N, M, K = map(int, input().split())
    ans1 = 0
    for d in range(1, N):
        ans1 = (ans1 + d * (N-d) * (M**2)) % MOD
    ans2 = 0
    for d in range(1, M):
        ans2 = (ans2 + d * (M-d) * (N**2)) % MOD
    ans = ((ans1 + ans2) * combination(M*N-2, K-2)) % MOD
    print(ans)


if __name__ == "__main__":
    main()
