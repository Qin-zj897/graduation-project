import math


def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def permutations_count(n, r):
    return math.factorial(n) // math.factorial(n - r)


n, m, k = map(int, input().split())
ans = 0
for i in range(m-1):
    for j in range(i+1, m):
        ans += combinations_count((n*m)-2, k-2) * n * n * (j - i)
for i in range(n-1):
    for j in range(i+1, n):
        ans += combinations_count((n*m)-2, k-2) * m * m * (j - i)
print(ans)