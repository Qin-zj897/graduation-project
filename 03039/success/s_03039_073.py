MOD = 10 ** 9 + 7
n, m, k = [int(item) for item in input().split()]


ans = 1
for i in range(2, k):
    ans = (ans * (n * m - i)) % MOD

for i in range(1, k - 2 + 1):
    ans = (ans * pow(i, MOD - 2, MOD)) % MOD

print((ans * (m ** 2 * n * (n ** 2 - 1) // 6 + n * (n - 1) * m
              * (m ** 2 - 1) // 6 + n * m * (m ** 2 - 1) // 6)) % MOD)


# pairs = []
# for i in range(1, n + 1):
#     for j in range(1, m + 1):
#         pairs.append((i, j))
#
# pairs.sort()
# print(pairs)
#
# brute = 0
# for i in range(len(pairs)):
#     for j in range(i + 1, len(pairs)):
#         brute += abs(pairs[i][0] - pairs[j][0]) + \
#                      abs(pairs[i][1] - pairs[j][1])
#
# print(ans * brute)
