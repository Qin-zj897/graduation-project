import math

if __name__ == '__main__':
    a,  b, x = map(int, input().split())
    if 2.0 * (b - x / a / a) <= b:
        print(math.degrees(math.atan(2.0 * (b - x / a / a) / a)))
    else:
        print(math.degrees(math.atan((a * b * b)/2.0/x)))

