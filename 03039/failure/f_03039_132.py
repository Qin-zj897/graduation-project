modout = lambda x: x%1000000007

N, M, K = map(int, input().split())
N2 = N**2
M2 = M**2
Xpattern = [x * (N-x)*M2 for x in range(N)]
Ypattern = [y * (M-y)*N2 for y in range(M)]
coef = 1
num = min(K-2, N*M-K)
for i in range(N*M-2, N*M-2-num, -1):
    coef *= i
for j in range(1, num+1):
    coef //= j


print(modout((sum(Xpattern)+sum(Ypattern))*coef))