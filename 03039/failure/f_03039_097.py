# https://atcoder.jp/contests/abc127/tasks/abc127_e

import itertools
from collections import Counter
from collections import defaultdict
import bisect
import math

# Calculate count of combination
def combination(n, r):
    if r == 0:
        return 1
    a = 1
    b = 1
    for i in range(r):
        a *= (n - i)
        if a % (i + 1):
            a //= (i + 1)
        else:
            b *= (i + 1)
    return a // b


def main():
    MOD = 10**9 + 7
    N, M, K = map(int, input().split())

    com = combination(N * M - 2, K - 2)
    # print(com)
    com = com % MOD
    # print(com)

    ans = 0
    for i in range(N*M-1):
        for j in range(i + 1, N*M):
            # print('{} {}'.format(i, j))
            iy = i // M
            ix = i % M

            jy = j // M
            jx = j % M
            ans += (abs(iy - jy) + abs(ix - jx)) * com
            ans = ans % MOD

    print(ans)



if __name__ == '__main__':
    main()
