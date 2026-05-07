from math import *
PI = pi

def main():
    A,B,X=map(int,input().split())
    theta=0
    if A*A*B/2<=X:
        theta=atan(2*(A*A*B-X)/(A*A*A))
    else:
        theta=atan(A*B*B/(2*X))
    print(theta*180/PI)

main()
