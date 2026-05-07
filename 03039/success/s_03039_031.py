def factorial_mod(n, mod):
    a = 1
    for i in range(1, n + 1):
        a *= i
        a %= mod
    return a


def comb_mod(n, k, mod):
    fact_n = factorial_mod(n, mod)
    fact_k = factorial_mod(k, mod)
    fact_n_k = factorial_mod(n - k, mod)
    return (fact_n * pow(fact_k, mod - 2, mod) * pow(fact_n_k, mod - 2, mod)) % mod


n, m, k = map(int, input().split())
s = 0
mod = 10 ** 9 + 7
for x in range(m):
    s += (m - x - 1) * (m - x) // 2 * n **2
for y in range(n):
    s += (n - y - 1) * (n - y) // 2 * m ** 2


print(s * comb_mod(n * m - 2, k - 2, mod) % mod)
