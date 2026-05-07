import collections as c
ip = lambda : map(int, input().split())

def cmb(n,r):
    if n==0:return 0
    else:
        r=min(r,n-r)
        result=1
        for i in range(n-r+1,n+1):
            result*=i
        for i in range(1,r+1):
            result//=i
        return result

# ##############
N,M,K = ip()
ans = 0
mod = int(1e9+7)
pattern = cmb(N*M-2,K-2)
for i in range(N*M):
    s,t = i//M,i%M
    '''for j in range(i,N*M):
        jpair = (j//N,j%N)
        cost = abs(ipair[0] - jpair[0]) + abs(ipair[1] - jpair[1])'''
    tmpS = s*(s+1)//2 + (N-s-1)*(N-s)//2
    cost = tmpS*M + N*(t*(t+1) + (M-1-t)*(M-t))//2
    #print(s,t,cost)
    ans += ((cost*pattern)//2)%mod
    ans %= mod
        
print(ans)