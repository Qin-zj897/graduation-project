n, a, b, c = map(int, input().split(' '))
import math
def permutations_count(n, r):
  return math.factorial(n) // math.factorial(n - r)

def reduce(p, q):
  p, q = int(p), int(q)
  common = math.gcd(p, q)
  return (p // common, q // common)

def lcm(x, y):
    return (x * y) // math.gcd(x, y)

if a >= 100 or b >= 100:
  ans = n
elif c <= 0:
  p = n*permutations_count(n, 0)
  q = 2**(n-1)
  p, q = reduce(p,q)
  for i in range(n+1, 2*n):
    perm = permutations_count(n, i-n)
    fr = i*perm
    ac = 2**(i-1)
    fr, ac = reduce(fr, ac)
    l = lcm(q, ac)
    if q <= ac:
      p *= l // q
      q = l
    else:
      fr *= l // ac
      ac = l
    p += fr
    p, q = reduce(p,q)
  p, q = reduce(p, q)
  i = 0
  m = 10**9+7
  while True:
    r = (p + i*m)/q
    print(r)
    if r == int(r):
      ans = int(r)
      break
    i += 1
print(ans)