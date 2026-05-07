
N,A,B,C = map(int, input().split())
A=A/100
B=B/100
N=N+1
arya = [[0]*N]*N


for i in range(N):
    for j in range(N):
        if i==0 and j==0:
            arya[i][j]=1
        elif i==0 and j!=0:
            arya[i][j] = arya[i][j-1]*A
        elif i!=0 and j ==0:
            arya[i][j] = arya[i-1][j]*B
        elif i==N-1 and j==N-1: arya[i][j]=arya[i-1][j]+arya[i][j-1]
        elif i==N-1 : arya[i][j]=arya[i-1][j]+arya[i][j-1]*A
        elif j==N-1 : arya[i][j]=arya[i-1][j]*B+arya[i][j-1]
        elif 1<=i and 1<=j: arya[i][j]=arya[i-1][j]*B + arya[i][j-1]*A

print(1/arya[N-1][N-1])
