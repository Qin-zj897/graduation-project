n, m, k = map(int, input().split())
t = n * m - 2
memo = 1

for i in range(k - 2):
    memo *= t
    memo = memo // (i + 1)
    t -= 1
nmemo = (memo % (10 ** 9 + 7)) * n * n % (10 ** 9 + 7)
mmemo = (memo % (10 ** 9 + 7)) * m * m % (10 ** 9 + 7)

answer = ((n*(n+1)) % (10 ** 9 + 7)*(n-1)*mmemo//6 % (10**9+7))
answer = (answer+(((m - 1) * m) % (10 ** 9 + 7)
                  * (m + 1)*nmemo // 6)) % (10 ** 9 + 7)

print(answer)
