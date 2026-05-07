
import math
A, B, X = map(int, input().split())
pi = 3.14159265359

if X == A*A*B:
    print("0")
elif X <= A*A*B/2:
    print(math.atan(A*B*B/2/X)/pi*180)
else:
    print(math.atan(2*(A*A*B-X)/(A*A*A))/pi*180)
