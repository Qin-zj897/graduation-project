#import sys

#input = sys.stdin.readline
import math
#import bisect

def sosuhante(n):
    for k in range(2, int(math.sqrt(n))+1):
        if n% k ==0:
            return False
    return True


def main():
#    h,w,a,b = map(int, input().split())
    a,b,x=map(int,input().split())


    t1=math.atan(2*(a*a*b-x)/a/a/a)
    t2=math.atan(b/a)
    t3=math.atan(a*b*b/2/x)

    if x>=a*a*b/2:
        print(math.degrees(t1))
    else:
        print(math.degrees(t3))




if __name__ == "__main__":
    main()