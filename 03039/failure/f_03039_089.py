n, m, k = map(int, input().split())
t = n * m - 2
memo = 1
answer = 0

for i in range(k - 2):
    memo *= t
    memo = memo // (i + 1)
    t -= 1

for i in range(1, n):
    answer = (answer+((1+i)*(i)//2)*m*m*memo % (10**9+7)) % (10**9+7)
for i in range(1, m):
    answer = (answer + ((i+1) * (i) // 2) * n * n*memo %
              (10 ** 9 + 7)) % (10 ** 9 + 7)

print(answer)
