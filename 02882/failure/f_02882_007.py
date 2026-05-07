A,B,X=map(int, input().split())
C=2*X/(A*B)
if C<=A:
    math.atan(B/C)*(180)/math.pi
    C=2*X/(A*B)
    print(math.atan(B/C)*(180)/math.pi)
else:
    C=2*Y/(A*A)
    Y=A*A*B-X
    C=2*Y/(A*A)
    print(math.atan(C/A)*(180)/math.pi)
    