import math
n,m,k = map(int,input().split())
mod = 10**9+7
memo = {}
def factorial(num):
    if num == 0:
        return 1
    if not num in memo:
        memo[num] = num*factorial(num-1)
    return memo[num]

temp = math.factorial(n*m-2)//(math.factorial(k-2)*math.factorial(n*m-4-k))
ans = 0
for d in range(1,n):
    ans += d*((n-d)*(m**2))*temp
for d in range(1,m):
    ans += d*((m-d)*(n**2))*temp
print(ans%mod)