
import math

def cmb(n, r):
    if n - r < r: r = n - r
    if r == 0: return 1
    if r == 1: return n

    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]

    for p in range(2,r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1,r,p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot

    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])

    return result

N, M, K = map(int, input().rstrip().split(' '))

comb = cmb(N*M-2, K-2) % 1000000007

cost = 0
for d in range(1, N):
    cost += d * M**2 * (N - d) * comb
    cost %= 1000000007
for d in range(1, M):
    cost += d * N**2 * (M - d) * comb
    cost %= 1000000007
    
print(cost)

