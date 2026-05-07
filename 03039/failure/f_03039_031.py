N, M, K = map(int, input().split())

import math

def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

NM = N * M
bai = (K * combinations_count(NM , K)) // (NM**2 - NM)

sum_cost = 0

ich_list = [(i, j) for i in range(1, N+1) for j in range(1, M+1)]

for p in range(len(ich_list) - 1):
    for q in range(p+1, len(ich_list)):
        i1, j1 = ich_list[p]
        i2, j2 = ich_list[q]
#         print(i1, j1, i2, j2)
#         print(abs(i2 - i1) + abs(j2 - j1))
        sum_cost += abs(i2 - i1) + abs(j2 - j1)
                
print((sum_cost * bai) % (10**9 + 7))