def mod_combination(mod: int, n: int, r: int) -> int:
  a, b, c = mod_factorials(mod, [n, r, n - r])

  return a * pow(b, mod - 2, mod) % mod * pow(c, mod - 2, mod) % mod

def mod_factorials(mod: int, N: [int]) -> [int]:
  result = [1] * len(N)
  tuples = sorted([(i, n) for i, n in enumerate(N)], key=lambda t: t[1])

  fact = 1
  i = 1
  for t in tuples:
    for j in range(i, t[1] + 1):
      fact = fact * j % mod
      i = i + 1

    result[t[0]] = fact

  return result


def main():
  MOD = 10 ** 9 + 7

  N, M, K = [int(i) for i in input().strip().split(' ')]
  c = mod_combination(MOD, N * M - 2, K - 2)

  X = 0
  Y = 0
  for i in range(M):
    X += (M - i) * i
  for i in range(N):
    Y += (N - i) * i

  total = (X * (N ** 2) + Y * (M ** 2)) % MOD * c % MOD
  print(total)


main()