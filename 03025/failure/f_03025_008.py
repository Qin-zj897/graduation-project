N, A, B, C = [int(i) for i in input().split()]
P = 0.01*max(A, B)
X = N*(1-P)/P
print(int((N+X)*(1-0.01*C)))