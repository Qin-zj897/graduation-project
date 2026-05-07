N, M, K = map(int, input().split())

import math

from operator import mul
from functools import reduce
def combinations_count(n, r):
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

ans_x = 0
c = combinations_count(N*M-2, K-2)
#for i in range(1, N+1):
#    ans_x += N*(N+1)//2-i*(i+1)//2-i*(N-i)
    #for j in range(i+1, N+1):
    #    ans_x += j-i
ans_x = (N*N*(N+1) + N*(N+1)*(2*N+1)//6 - (2*N+1)*N*(N+1)//2)//2
ans_x = M*M*ans_x*c

ans_y = 0
c = combinations_count(N*M-2, K-2)
#for i in range(1, M+1):
#    ans_y += M*(M+1)//2-i*(i+1)//2-i*(M-i)
    #for j in range(i+1, M+1):
    #    ans_y += j-i
ans_y = (M*M*(M+1) + M*(M+1)*(2*M+1)//6 - (2*M+1)*M*(M+1)//2)//2
ans_y = N*N*ans_y*c
ans = (ans_x+ans_y)%(10**9+7)
print(ans)