n,m,k=map(int,input().split())
import math
def f(x): return math.factorial(x)
def c(x,y): return f(x)//(f(x-y)*f(y))
print(((n**2*(m-1)*m*(m+1)//6+m**2*(n-1)*n*(n+1)//6)*c(m*n-2,k-2))%(10**9+7))