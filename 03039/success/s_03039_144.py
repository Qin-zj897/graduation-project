def inverse(a, mod):
    while(a < 0):
        a += mod
    a %= mod
    p = mod
    ya = 0
    yp = 1
    while (a > 0):
        n = p // a
        r = p % a
        yr = ya - n * yp
        ya = yp
        yp = yr
        p = a
        a = r
    return ya % mod if ya >= 0 else ya + mod

mod = 1000000007

n, m, k = map(int, input().split())

inv = 1
for i in range(1, k-1):
    inv = inv * i % mod
inv = inverse(inv, mod)

combination = inv
for i in range(n*m-k+1, n*m-1):
    combination = combination * i % mod


xn = n * (n + 1) * (n - 1) // 6 % mod * m % mod * m % mod
yn = m * (m + 1) * (m - 1) // 6 % mod * n % mod * n % mod
ans = (xn + yn) * combination % mod
print(ans)

