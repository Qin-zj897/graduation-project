import itertools
N,M,K = map(int, input().split())
ans = 0
x = list(range(1,N+1))
y = list(range(1,M+1))
xy =  list(itertools.product(x,y))
NM = N*M
for i in range(NM):
  for j in range(i+1,NM):
    ans += abs(xy[i][0]-xy[j][0]) +  abs(xy[i][1]-xy[j][1])

print(ans % (10**9 + 7))    