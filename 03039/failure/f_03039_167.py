N, M, K = (int(x) for x in input().split())

def cell_dist_sum(i,j):
    x = (i*(i+1)//2+(N-i-1)*(N-i)//2)*M
    y = (j*(j+1)//2+(M-j-1)*(M-j)//2)*N
    return x+y

def nCm(n,m):
    if m < 0 or m > n:
        return 0
    nf = factorial(n)
    nm = factorial(m)
    nm2 = factorial(n-m)
    return (nf//nm//nm2) % 1000000007

def factorial(n):
    ans = 1
    for i in range(n):
        ans *= (i+1)
    return ans

c_sum = 0
for i in range(N):
    for j in range(M):
        c_sum += cell_dist_sum(i,j)//2
        c_sum = c_sum % 1000000007
c_sum *= nCm(N*M-2, K-2)
c_sum = c_sum % 1000000007

print(c_sum)
