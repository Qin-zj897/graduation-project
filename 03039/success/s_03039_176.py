def mfactorial(n: int, mod: int) -> int:
  cache = mfactorial.__cache if hasattr(mfactorial, '__cache') else [1, 1]
  n_len = len(cache)
  n_new = n - n_len + 1
  if n_new > 0:
    cache = cache + ([0] * n_new)
    for i in range(n_len, n + 1):
      cache[i] = cache[i - 1] * i % mod
    mfactorial.__cache = cache

  return mfactorial.__cache[n]


def cmb(n, r, mod):
  result = mfactorial(n, mod)
  result *= pow(mfactorial(r, mod), mod - 2, mod) % mod
  result %= mod
  result *= pow(mfactorial(n - r, mod), mod - 2, mod) % mod
  result %= mod

  return result


def main():
  MOD = 10 ** 9 + 7

  N, M, K = [int(i) for i in input().strip().split(' ')]
  c = cmb(N * M - 2, K - 2, MOD)

  X = 0
  Y = 0
  for i in range(M):
    X += (M - i) * i
  for i in range(N):
    Y += (N - i) * i

  total = (X * (N ** 2) + Y * (M ** 2)) % MOD * c % MOD
  print(total)


main()