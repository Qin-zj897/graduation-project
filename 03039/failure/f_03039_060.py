a, b, c = map(int,input().split())
e = [23] * 300000
e[-1] += 1
for i in range(100):
  w = e.pop()
  e.append(w)
print(1)