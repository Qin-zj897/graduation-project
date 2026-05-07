from itertools import prodcut
n,m,k = map(int, input().split())
a, b = list(range(n)), list(range(m))
l = product(a,b)