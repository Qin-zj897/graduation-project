from fractions import Fraction as F
I=lambda:map(int,input().split())
mod=1000000007
N,A,B,C=I()
a=F(A,100); b=F(B,100); c=F(C,100)
f=N*(a**N)/((1-c)**N)/(1-c)*((b**N)/(c**N)-1)/((b/c)-1)
g=N*(b**N)/((1-c)**N)/(1-c)*((a**N)/(c**N)-1)/((a/c)-1)
print(f+g)