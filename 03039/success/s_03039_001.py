n,m,k=map(int,input().split())
ans=0
INF=10**9+7
if n<=m:
    M=(m-1)*m*(2*m-1)//3
    for i in range(1,n+1):
        for j in range(i,n+1):
            if i==j:
                ans=ans+m*m*(m-1)//2-(m-1)*m*(2*m-1)//6
            else:
                d=j-i
                ans=ans+d*m+(m+d)*m*(m-1)-M
            ans=ans%INF
else:
    (n,m)=(m,n)
    M=(m-1)*m*(2*m-1)//3
    for i in range(1,n+1):
        for j in range(i,n+1):
            if i==j:
                ans=ans+m*m*(m-1)//2-(m-1)*m*(2*m-1)//6
            else:
                d=j-i
                ans=ans+d*m+(m+d)*m*(m-1)-M
            ans=ans%INF
import math
ans=(math.factorial(n*m-2)//(math.factorial(n*m-k)*math.factorial(k-2)))*ans
ans=ans%INF
print(ans)
