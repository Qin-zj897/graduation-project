N,M,K=map(int,input().split())
P=1000000007
import math
ans=0
for A in [N,M]:
    for i in range(1,A):
        ans = (ans+(N^M^A)**2*i*(i+1)//2)%P
print(ans*math.factorial(N*M-2)*pow(math.factorial(K-2),P-2,P)*pow(math.factorial(N*M-K),P-2,P)%P)
