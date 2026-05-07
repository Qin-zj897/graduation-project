N, M, K = map(int, input().split())
P = int(1e9+7)
N -= 1
M -= 1

def comb(n, r):
    if n < r or r < 0:
        return 0
    nume = 1
    deno = 1
    for i in range(1, r + 1):
        nume = (nume * (n - i + 1)) % P
        deno = (deno * i) % P
    deno_inv = pow(deno, P-2, P)
    return (nume*deno_inv)%P

ans = 0

def sum_man_dist_ur(u, r):
    ret = 0
    u, r = min(u, r), max(u, r)
    if u == 0:
        ret =  r * (r + 1) // 2
    else:
        ret += (u * (u + 1) * (u - 1)) // 3
        ret += (u+1)*(r-u+1)*(r+u)//2
        ret += u * (u + 1) * (u + 3 * r + 2) // 6
    return ret%P

def sum_man_dist(x, y):
    ret = 0
    ret += sum_man_dist_ur(N - x, M - y)
    ret = (ret + sum_man_dist_ur(N - x, y)) % P
    ret = (ret + sum_man_dist_ur(x, M - y)) % P
    ret = (ret + sum_man_dist_ur(x, y)) % P
    ret -= (N-x)*(N-x+1)//2
    ret -= (x)*(x+1)//2
    ret -= (M-y)*(M-y+1)//2
    ret -= (y) * (y + 1) // 2
    return ret%P

ans = 0
c = comb((N+1)*(M+1)-2, K-2)
for x in range(N + 1):
    for y in range(M + 1):
        ans = (ans + c * sum_man_dist(x, y)) % P
print(ans*pow(2, P-2, P)%P)
