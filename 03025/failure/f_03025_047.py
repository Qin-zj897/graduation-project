
N,A,B,C = list(map(int,input().split()))

P = N*((A+B)**2)*C
Q = A*B*100

def gcd(a, b):
    if b == 0:
      return a
    return gcd(b,a%b)

g=gcd(P,Q)

P=P//g
Q=Q//g

for i in range(Q):
    if (P+ ((10**9) +7)*i)%Q==0:
        R = (P+ ((10**9) +7)*i)//Q
        break
print(R)