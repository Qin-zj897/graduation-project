from fractions import Fraction

n,a,b,c = map(int,input().split())

def nPowSum(m,x,y):
    sum = Fraction(0,1)
    for i in range(m + 1):
        sum = sum + ((x ** i) * (y ** (m - i)))
    return sum


a_p = Fraction(a,100)
b_p = Fraction(b,100)
c_p = Fraction(c,100)
s = Fraction(0,1)
M = ((a_p ** n) * nPowSum(n - 1,b_p,c_p)) + ((b_p ** n) * nPowSum(n - 1,a_p,c_p))
#print(M)
if n >= 2:
    for i in range(n - 2):
        s = s + ((a_p ** n) * nPowSum(i,b_p,c_p))
        s = s + ((b_p ** n) * nPowSum(i,a_p,c_p))
        s = s * (n + i)

#print(s)

s = s + M * Fraction(2 * n - 1,1) / (1 - c_p)
s = s + M * c_p / ((1 - c_p) ** 2)

#print(s)

p = s.numerator
q = s.denominator
k = 0

while True:
    if p % q == 0:
        print(int(p / q))
        break
    else:
        k = k + 1
        p = p + 1000000007
