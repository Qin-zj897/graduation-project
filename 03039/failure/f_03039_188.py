import math

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

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


                                                  
    
