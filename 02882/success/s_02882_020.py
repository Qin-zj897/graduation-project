from math import atan2, degrees
A, B, X = map(int, input().split())
S = X / A


if S >= A * B / 2:
    h = (2 * (A * B - S)) / A
    print(degrees(atan2(h, A)))

else:
    w = (2 * S) / B
    print(degrees(atan2(B, w)))