N,M,K=map(int,input().split())
P=1000000007
import math
ans=M**2*(N-1)*N*(N+1)//6
ans+=N**2*(M-1)*M*(M+1)//6
print(ans*math.factorial(N*M-2)*pow(math.factorial(K-2),P-2,P)*pow(math.factorial(N*M-K),P-2,P)%P)
