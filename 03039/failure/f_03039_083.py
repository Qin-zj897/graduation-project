import math
N,M,k = map(int,input().split())
p = N*M
PP = 1
for i in range(p-k+1,p-1):
    PP *= i
P = PP//math.factorial(k-2)
    
Q = N*(N-1)*(N+1)*M*M//6
R = M*(M-1)*(M+1)*N*N//6
print((P*(Q+R))%(10**9+7))