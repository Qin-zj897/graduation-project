n, m, k = map(int,input().split())
mod = 10**9 + 7
l = 3*10**5 + 3

M = [1]  # i!のmod
mul = 1
for i in range(1, l):
    mul = (mul * i) % mod
    M.append(mul)

def pow(x, y, mod):  # x**y の mod を返す関数
    ans = 1
    while y > 0:
        if y % 2 == 1:
            ans = (ans * x) % mod
        x = (x**2) % mod
        y //= 2
    return ans

def inv(x, mod):  # x の mod での逆元を返す関数
    return pow(x, mod-2, mod)

# print(inv(8,13))

MI = [0] * (l-1) +[inv(M[l-1], mod)]  # i!の逆元
for i in range(l-2, -1, -1):
    MI[i] = MI[i+1] * (i+1) % mod

def C(x, y):  # コンビネーション（組合せ）
    if y < 0 or y > x:
        return 0
    elif x > l:  # O(min(y, x-y))
        y = min(y, x-y)
        ans = 1
        for i in range(x, x-y, -1):
            ans = (ans * i) % mod
        return (ans * MI[y]) % mod
    else:  # O(1)
        ans = M[x]
        ans = (ans * MI[y]) % mod
        return (ans * MI[x-y]) % mod

def P(x, y):  # パーミュテーション（順列）
    if y < 0 or y > x:
        return 0
    elif x > l:  # O(min(y, x-y))
        y = min(y, x-y)
        ans = 1
        for i in range(x, x-y, -1):
            ans = (ans * i) % mod
        return ans % mod
    else:  # O(1)
        ans = M[x]
        return (ans * MI[x-y]) % mod

a = 0
for i in range(1, m):
    a = (a + n*n*(m-i)*i) % mod
for i in range(1, n):
    a = (a + m*m*(n-i)*i) % mod

print((a * C(n*m-2, k-2)) % mod)