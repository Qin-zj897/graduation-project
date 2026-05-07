n, a, b, c = map(int, input().split())
ans = 0
i = 1
k = 1
while:
  ans += (a+b)*k*i
  k = k * c
  if i == 2*n-1:
    break
print(ans)