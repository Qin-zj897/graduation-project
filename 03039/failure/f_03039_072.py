import math

dp = [0 for _ in range(2*10**5+1)]
f = math.factorial
def fact(n):
    if n == 0:
        return 1
    if dp[n] != 0:
        return dp[n]
    else:
        fn = f(n)
        dp[n] = fn
        return fn

def nCr(n,r):
    return fact(n) // fact(r) // fact(n-r)

def distancesum(arr, n): 
      
    # sorting the array. 
    arr.sort() 
      
    # for each point, finding  
    # the distance. 
    res = 0
    sum = 0
    for i in range(n): 
        res += (arr[i] * i - sum) 
        sum += arr[i] 
      
    return res 

flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]

div = 10**9 + 7

N, M, K = map(int, input().split())
repeat = nCr(N*M-2, K-2)
f_row = flatten([[i for _ in range(M)] for i in range(N)])
f_col = flatten([[i for i in range(M)] for _ in range(N)])

manhsum = distancesum(f_row, len(f_row)) + distancesum(f_col, len(f_col))
print(((repeat % div) * (manhsum % div))%div)
