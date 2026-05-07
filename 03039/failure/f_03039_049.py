N, M, K = map(int, input().split())

DIV = 10**9+7

def comb(l, r):
    if l == 0:
        return 1
    val = 1
    for i in range(r, r-l, -1):
        val *= i
    for i in range(l, 0, -1):
        val /= i
    return val

def calc_rec_num(N, M, xi, xj, yi, yj):
    cnt = 0
    cnt += N - yi
    cnt += M - yj
    # rotate
    cnt *= 2
    return cnt


p = comb(K-2, N*M-2)
ans = 0
for i in range(N):
    for j in range(M):
        if i == 0 and j == 0:
            continue
        d = i+j
        ans += d * calc_rec_num(N, M, 0, 0, i, j) * p

print(ans % DIV)
