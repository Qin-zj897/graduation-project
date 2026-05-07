#21:50
h,w,k = map(int,input().split())
mod = 10 ** 9 + 7
def inv(x):
  y = 1
  while x != 1:
    y *= mod//x + 1
    y %= mod
    x = x - mod%x
  return y
others = 1
for i in range(k-2):
  others *= h*w-2-i
  others *= inv(i+1)
  others %= mod
hh = 0
for i in range(1,h):
  hh += i * (h-i)
  hh %= mod
ww = 0
for i in range(1,w):
  ww += i * (w-i)
  ww %= mod
ans = others * ( h * h * ww + w * w * hh )
print(ans % mod)