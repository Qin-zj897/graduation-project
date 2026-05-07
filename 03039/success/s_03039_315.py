def power_mod(a,b,mod=10**9+7):
    i,temp,box=0,b,[]
    while(2**i<=b):
        i+=1
    for j in range(i-1,-1,-1):
        box=[[j,temp//2**j]]+box
        temp-=2**j*(temp//2**j)
    box[0].append(a)
    ans=box[0][1]*a%mod
    for j in range(1,i):
        box[j].append(box[j-1][2]**2%mod)
        if box[j][1]==1:
            ans=(ans*box[j][2])%mod
    return ans
def n_func(n,mod=10**9+7):
    ans=1
    for i in range(1,n+1):
        ans=(ans*i)%mod
    return ans
def nPr(n,r,mod=10**9+7):
    ans=n_func(n-r,mod)
    ans=power_mod(ans,mod-2,mod)
    return ans*n_func(n,mod)%mod
def nCr(n,r,mod=10**9+7):
    ans=n_func(n-r,mod)*n_func(r,mod)%mod
    ans=power_mod(ans,mod-2,mod)
    return ans*n_func(n,mod)%mod

N,M,K=map(int,input().split())
keisuu=nCr(N*M-2,K-2)
mod=10**9+7

sum_=0
for i in range(N):
    a=min(abs(i),abs(N-i-1))
    b=max(abs(i),abs(N-i-1))
    sum_+=(M**2)*((a*(a+1)//2)+(b*(b+1)//2))
for i in range(M):
    a=min(abs(i),abs(M-i-1))
    b=max(abs(i),abs(M-i-1))
    sum_+=(N**2)*((a*(a+1)//2)+(b*(b+1)//2))
print((keisuu * (sum_//2))%mod)