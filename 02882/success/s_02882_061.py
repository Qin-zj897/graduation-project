# abc 144 d
import math
read = input 
rn = lambda :list(map(int, read().split()))
def reverse_range(r):return range(r-1, -1, -1)
a, b, x = rn()
thres = a * a * b
if x*2 >= thres:
    j = (2*x) / (a*a)
    j -= b
    tan = (b-j) / a
    ans = math.atan(tan)
    ans = math.degrees(ans)
    print(ans)
    pass
else :
    j = (2*x) / (a*b)
    tan = j / b
    ans = math.atan(tan)
    # math.radians()
    print(90 - math.degrees(ans))