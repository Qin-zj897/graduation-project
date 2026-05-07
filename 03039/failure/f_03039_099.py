import math
N,M,K = map(int,input().split())
pat = 1

ans_x,ans_y = 0,0
for i in range(M):
    ans_x =(ans_x + i*(M-i))%(10**9+7)
for i in range(N):
    ans_y =(ans_y + i*(N-i))%(10**9+7)
ans_x = (ans_x*(N**2))%(10**9+7)
ans_y = (ans_y*(M**2))%(10**9+7)
ans = (ans_x+ans_y)%(10**9+7)
ans = (ans*pat)%(10**9+7)
print(ans)