from fractions import Fraction


n, a, b, c = list(map(int, input().split()))
x = 10**9 + 7
a = a/100
b = b/100
c = c/100
kitaiti = 0
for i in range(100000):
    kitaiti += (i+n)*((c**i)*(a**n) + (c**i)*(b**n))
print(round(kitaiti, 10).as_integer_ratio())
p, q = round(kitaiti, 10).as_integer_ratio()[0], round(kitaiti, 3).as_integer_ratio()[1]

for i in range(x-1):
    print(i * q % x == p)
    if  i * q % x == p:
        print(i)
        break
