#E問題
N,M,K = map(int,input().split())
mod = 1000000007

#factorial(階乗)
def factorial(n):
    factorial_list=[]
    for i in range(n+1):
        if i == 0:
            factorial_list.append(1)
        elif i == 1:
            factorial_list.append(1)
        else:
            factorial_list.append(factorial_list[-1]*i)
    return factorial_list[-1]
#combination(組み合わせC)
def comb(comb1,comb2):
    x=min(comb1,comb2)
    y=max(comb1,comb2)
    bunshi=factorial(y)
    bunbo=factorial(x)*factorial(y-x)
    return bunshi//bunbo

C = comb(N*M-2,K-2)


X = 0
for i in range(N-1):
    d = i+1
    X+=(N-d)*M*M*d
    X%=mod
Y = 0
for i in range(M-1):
    d = i+1
    Y+=(M-d)*N*N*d
    Y%=mod
print(C*(X+Y)%mod)