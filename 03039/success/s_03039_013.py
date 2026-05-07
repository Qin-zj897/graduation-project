N,M,K = map(int, input().split())
P=1000000007
MPL = 1
f = 1
for i in range(1, N*M-1):
    MPL = MPL * i % P
    MPL %= P
    if i == K - 2:
        f *= pow(MPL, P - 2, P)
    if i == N*M - K:
        f *= pow(MPL, P - 2, P)
MPL = MPL * f % P
ans = 0
for A in [N, M]:
    width = N ^ M ^ A
    for i in range(1, A):
        ans = (ans + width * width * i * (i + 1) // 2) % P
print(ans * MPL % P)