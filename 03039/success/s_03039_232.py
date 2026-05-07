import math
import sys
sys.setrecursionlimit(10 ** 6 + 10)
from collections import defaultdict

from math import factorial

def combinations_count(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))

def permutations_count(n, r):
    return math.factorial(n) // math.factorial(n - r)



def main():
    mod = 10 ** 9 + 7
    high, width, num = map(int, input().split())
    toori = combinations_count(high * width - 2, num - 2)

    kyori = high + width - 2

    ans = 0
    for now_kyori in range(1, kyori + 1):
        kari = 0
        if now_kyori < high:
            kari += width * width * (high - now_kyori)
        if now_kyori < width:
            kari += high * high * (width - now_kyori)
        ans += now_kyori * kari

        if now_kyori >= high and now_kyori >= width:
            break

    ans = ans * toori % mod

    print(ans)



if __name__ == '__main__':
    main()
