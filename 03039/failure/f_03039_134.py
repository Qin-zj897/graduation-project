import math

inf = 7 + (10 ** 9)

def combinations_count(n, r):
    return (math.factorial(n) // (math.factorial(n - r) * math.factorial(r))) % inf

n, m, k = map(int, input().split())
ans = 0
comb = ((combinations_count((n*m)-2, k-2)))
n_2 = (n * n * comb) % inf
m_2 = (m * m * comb) % inf
min_n_m = min(n, m)
for i in range(1, min_n_m):
    ans += (( n_2 * (i)) * (m - i)) % inf
    ans = ans % inf
for i in range(1, n):
    ans += (m_2 * (i) * (n - i)) % inf
    ans = ans % inf
ans = ans % inf
print(ans)