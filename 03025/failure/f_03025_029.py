from fractions import Fraction
n, a, b, c = list(map(int, input().split()))

a = a * 0.01
b = b * 0.01
c = c * 0.01

e = 0
for k in range(1, 2*n + 1):
    for i in range(k - n):
        e += (a * (n-1) + b * i + c *(k-n-i)) * k * 2
        
print(e)
frac = Fraction(e).limit_denominator(1000)
p = frac.numerator
q = frac.denominator
mod = 2 * n
for num in range(mod):
    if num * q % mod:
        print(num)