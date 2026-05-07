N, M, K = map(int, input().split())

DIV = 10**9+7

def comb(l, r):
    if l == 0:
        return 1
    val = 1
    for i in range(r, r-l, -1):
        val *= i
    for i in range(l, 0, -1):
        val //= i
    return val


#p = comb(K-2, N*M-2)
#print(K-2, N*M-2, p)
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

print((ans*p)%DIV)
