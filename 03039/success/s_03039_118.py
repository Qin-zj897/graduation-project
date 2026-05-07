# Enter your code here. Read input from STDIN. Print output to STDOUT

MOD=10**9+7
fact=[1]
infact=[1]
temp=1
intemp=1
for i in xrange(1,10**6):
    temp*=i
    temp%=MOD
    fact+=[temp]
    intemp*=pow(i,MOD-2,MOD)
    intemp%=MOD
    infact+=[intemp]
    

def binom(a,b):
    up=fact[a]
    down=(infact[b]*infact[a-b])%MOD
    ans=(up*down)%MOD
    return ans


N,M,k=list(map(int,raw_input().strip().split(' ')))
if N==1 or M==1:
    if M<N:
        N,M=M,N
    ans=(binom(M*N,k)*binom(k,2))%MOD
    ans*=binom(M+1,3)
    ans%=MOD
    ans*=pow(binom(M*N,2),MOD-2,MOD)
    ans%=MOD
    print(ans)

else:       


    ans=(binom(M*N,k)*binom(k,2))%MOD
    ans*=((M*M*binom(N+1,3))%MOD+(N*N*binom(M+1,3))%MOD)%MOD
    ans*=pow(binom(M*N,2),MOD-2,MOD)
    ans%=MOD
    print(ans)
    
    
    

