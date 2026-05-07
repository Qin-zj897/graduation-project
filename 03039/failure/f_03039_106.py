a, b, c = map(int,input().split())
e = [23] * 300000
e[-1] += 1
for i in range(100000):
  if i > 10:
    continue
  w = min(e)
print(1)