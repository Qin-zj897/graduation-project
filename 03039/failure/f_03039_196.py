import math
N,M,K=map(int,input().split())
xs=0
ys=0
inf=10**9+7
xs=(M*(M+1)*(2*M+1) - 3*M*(M+1))//12
ys=(N*(N+1)*(2*N+1) - 3*N*(N+1))//12
#for i in range(1,M):
	#xs+=i*(M-i)
	#xs = xs % inf
#for i in range(1,N):
	#ys+=i*(N-i)
	#ys = ys % inf
xs*=(N**2)
xs = xs % inf
ys*=(M**2)
ys = ys % inf

def comb(n,r):
	return math.factorial(n) // (math.factorial(n-r)*math.factorial(r))

def comb_new(n,r):
	inf = 10**9+7
	loop = n//inf
	amari = n % inf
	if loop:
		y = 1
		for i in range(r):
			y *= n-i
		return y//math.factorial(r)
	else:
		y = 1
		for i in range(r):
			y *= n-i
		return y//math.factorial(r)


#c = comb(N*M-2,K-2) % inf
c = comb_new(N*M-2,K-2) % inf
print(((xs+ys)*c)%inf)