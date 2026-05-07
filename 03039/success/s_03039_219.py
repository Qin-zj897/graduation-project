N,M,K=map(int,input().split())
P=1000000007
import math
print(M*N*(M+N)*(M*N-1)//6*math.factorial(N*M-2)*pow(math.factorial(K-2),P-2,P)*pow(math.factorial(N*M-K),P-2,P)%P)
