from math import factorial

def combinations_count(n, r):
    
#     return factorial(n) // (factorial(n - r) * factorial(r))

    result = 1
    for i in range(n-r+1,n+1):
        result = result * i % div_num

    bunbo = 1
    for i in range(1,r+1):
        bunbo *= i

    return result//bunbo

N,M,K = map(int,input().split())

div_num = 10**9 + 7

cost = 0
def func(n,m,k):
    result = 0
    for d in range(1,n):
        select_2cell_cnt = (n-d)*(m**2)
        result += d * select_2cell_cnt * combinations_count(n*m-2,k-2)
        result = result % div_num
    
    return result % div_num

cost += func(N,M,K)
cost += func(M,N,K)

print(cost % div_num)