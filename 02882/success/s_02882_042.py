import math

a, b, x = map(int, input().split())

if pow(a,2) * b == x:
    print(0)
elif pow(a,2) * b / 2 <= x:
    print(90 - math.degrees(math.atan((a / 2) / (b - x / pow(a,2)))))
else:
    print(90 - math.degrees(math.atan((2 * x / (a * b)) / b)))

'''
left = 0
right = math.pi / 2
for i in range(10**5):
    mid = (left + right) / 2
    water = b * b * math.tan(math.pi / 2 - mid) * 0.5 * a
    if water > x:
        left = mid
    elif water < x:
        right = mid
    else: break
print(i, water, x)
print(math.degrees(mid))
'''