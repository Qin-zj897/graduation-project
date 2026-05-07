import sys
import math


def main():
    a, b, x = map(int, sys.stdin.readline().strip().split())

    s = x / a
    if s > a * b / 2:
        ans = math.atan(1 / (2 / a * (b - s / a)))
    else:
        ans = math.atan(2 * s / (b ** 2))
        
    print(90 - math.degrees(ans))


if __name__ == '__main__':
    main()