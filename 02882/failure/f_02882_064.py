# ABC144d


def main():
    import sys
    import math
    input = sys.stdin.readline
    sys.setrecursionlimit(10**6)

    a, b, x = map(int, input().split())
    if a*b*b/2 > x:
        print(math.degrees(math.atan(b*b*a/2/x)))
        exit(0)
    print(math.degrees(math.atan((b-x/a/a)/a*2)))


if __name__ == '__main__':
    main()