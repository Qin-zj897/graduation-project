from itertools import product, combinations
N,M,K=map(int, raw_input().split())


mod=10**9+7
fact, inv_fact = [1], [1]
fact_tmp = 1
for i in range(1, N*M+1):
    fact_tmp *= i
    fact_tmp %= mod
    fact.append(fact_tmp)
    inv_fact.append(pow(fact_tmp, mod-2, mod))
 
 
def ncr(n,r):
	if n < 0 or r < 0 or n < r:	return 0
	else:	return (fact[n] * inv_fact[r] * inv_fact[n-r]) %mod


torisuu=ncr(N*M-2,K-2)

ans=0
for i in range(N):
	for j in range(M):
		l_sum=(1+j)*j/2*N		#left
		r_sum=(1+(M-j-1))*(M-j-1)/2*N	#right	
		u_sum=(1+i)*i/2*M		#upper
		d_sum=(1+(N-i-1))*(N-i-1)/2*M	#down
		ans+=(l_sum+r_sum+u_sum+d_sum)*torisuu
		ans%=mod

print ( ans*pow(2,mod-2,mod) ) %mod