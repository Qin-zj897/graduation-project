import math
def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


n, m, k = map(int, input().split())
cmb = combinations_count(n * m - 2, k - 2)
cmb %= 10 ** 9 + 7
ans = 0
for i in range(1, n):
    ans += i * cmb * m * m * (n - i)
    ans %= 10 ** 9 + 7

for i in range(1, m):
    ans += i * cmb * n * n * (m - i)
    ans %= 10 ** 9 + 7

print(ans % (10 ** 9 + 7))