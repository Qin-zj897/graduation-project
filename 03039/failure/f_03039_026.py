from functools import reduce
from operator import mul


def cmb2(n, r):
  r = min(n - r, r)
  if r == 0:
    return 1
  over = reduce(mul, range(n, n - r, -1))
  under = reduce(mul, range(1, r + 1))

  return over // under


def main():
  MOD = 1000000007

  N, M, K = [int(i) for i in input().strip().split(' ')]
  c = cmb2(N * M - 2, K - 2) % MOD
  print(c)

  X = 0
  Y = 0
  for i in range(M):
    X += (M - i) * i
  for i in range(N):
    Y += (N - i) * i

  total = (X * (N ** 2) + Y * (M ** 2)) % MOD * c % MOD
  print(total)


main()