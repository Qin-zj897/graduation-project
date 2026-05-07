import sys
from itertools import combinations
try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def main():
    input = sys.stdin.readline
    MOD = 10**9 + 7
    N, M, K = map(int, input().split())
    # The number of times that pieces are located on certain two tiles.
    num = int(comb(N*M-2, K-2, exact=True))

    ans = 0
    # Column-wise distance
    for a, b in combinations(range(M), 2):
        ans += abs(a - b) * N**2 * num
        ans %= MOD

    # Row-wise distance
    for a, b in combinations(range(N), 2):
        ans += abs(a - b) * M**2 * num
        ans %= MOD

    return ans


if __name__ == '__main__':
    print(main())
