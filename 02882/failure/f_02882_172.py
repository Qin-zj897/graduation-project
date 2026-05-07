import math

a, b, x = map(int,input().split())
h = x/(a**2) #真っ直ぐ入れた時の高さ
s = b*h #四角形の面積
print(h, s)

if h*2 <= b: #こぼれる時に三角形
    ans = math.degrees(math.atan(b**2/2/s))

elif h*2 < b: #こぼれる時に台形
    print(1)
print(ans)