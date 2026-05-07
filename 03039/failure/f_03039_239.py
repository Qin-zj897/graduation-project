import math

inf = 7 + (10 ** 9)

def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


def permutations_count(n, r):
    return math.factorial(n) // math.factorial(n - r)


n, m, k = map(int, input().split())
ans = 0
for i in range(1, m-1):
    ans += (combinations_count((n*m)-2, k-2) * n * n * (i)) * (m - i)
    ans = ans % inf
for i in range(1, n-1):    
    ans += combinations_count((n*m)-2, k-2) * m * m * (i) * (n - i)
    ans = ans % inf
ans = ans % inf
print(ans)