N, M, K = map(int, input().split())

MOD = 10**9+7
fact = [1 for i in range(N+1)]
for i in range(2, N+1):
    fact[i] = fact[i-1] * i

def comb(n, r):
    return fact[n] // fact[n-r] // fact[r]

p = comb(K-2, N*M-2)
ans = 0
for i in range(N):
    for j in range(M):
        if i == 0 and j == 0:
            continue
        d = i+j
        #print(i, j, calc_rec_num(N, M, 0, 0, i, j))
        cnt = (N-i) * (M-j)
        if i != 0 and j != 0:
            cnt *= 2
        ans += d * cnt
        #if ans > DIV:
        #    ans %= DIV
        #print(ans)

print((ans*p)%MOD)
