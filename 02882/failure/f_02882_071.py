import math
a, b, x = map(int, input().split())
# ans1 = 0
# ans2 = 0
# if a > b:
#     ans = math.atan2(2 * x, a * a * a)
#     ans *= 180 / math.pi
#     ans = ans1
# else:
#     ans1 = math.atan2(2 * x, b * b * b)
#     ans1 *= 180 / math.pi
#     ans1 = 90 - ans1
#
#     ans2 = math.atan2(2 * (a * a * b - x), a * a * a)
#     ans2 *= 180 / math.pi
#     ans = max(ans1, ans2)
# # if a * math.tan(ans3) > b:
# #     ans3 = ansA
# # else:
# print(math.atan2(2 * x, a * a * a))

if x >= a * a * b / 2:
    ans = math.atan2(2 * (a * a * b - x), a * a * a)
    ans *= 180 / math.pi
# elif a > b:
#     ans = math.atan2(2 * x, a * a * a)
#     ans *= 180 / math.pi
else:
    ans = math.atan2(2 * x, b * b * b)
    ans *= 180 / math.pi
    ans = 90 - ans
print(ans)