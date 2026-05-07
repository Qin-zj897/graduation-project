n, m, k = map(int, input().split())
MOD = (10**9) + 7


S = 0

def stick(n):
    return n * (n+1) // 2

def calc(w, h):
    L = w * (w-1) // 2
    R = L + w * (h-1)
    return (L + R) * h // 2

for i in range(n):
    for j in range(m):
        S += calc(i+1, j+1)
        S += calc(i+1, m-j)
        S += calc(n-i, j+1)
        S += calc(n-i, m-j)
        S -= stick(i)
        S -= stick(j)
        S -= stick(n-i-1)
        S -= stick(m-j-1)
        S %= MOD
        
if S % 2 == 0:S //= 2
else: S = (S + MOD) // 2
S %= MOD
if S < 0:S += MOD

def fact(x):
    res = 1
    for i in range(1, x+1):
        res = res * i % MOD
    return res


def rev(x):
    return pow(x, MOD-2, MOD)

def nCk(n, k):
    return fact(n) * rev(fact(k)) % MOD * rev(fact(n-k)) % MOD

print(S * nCk(n*m-2, k-2) % MOD)