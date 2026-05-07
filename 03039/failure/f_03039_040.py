import math

inf = 7 + (10 ** 9)

def combinations_count(n, r):
    return (math.factorial(n) // (math.factorial(n - r) * math.factorial(r))) % inf

n, m, k = map(int, input().split())
ans = 0
comb = ((combinations_count((n * m) - 2, k - 2)))
n_2 = (n * n * comb) % inf
m_2 = (m * m * comb) % inf
k_m = (m * (m - 1) * (m + 1)) % inf
k_n = (n * (n - 1) * (n + 1)) % inf
ans = (k_m * n_2) + (k_n * m_2)
ans = ans % inf
print(ans)