# ABC144D - Water Bottle
from math import atan, degrees


def main():
    A, B, X = map(int, input().split())
    if A ** 2 * B >= 2 * X:
        C = 2 * X / (A * B)
        ans = degrees(atan(B / C))
    else:
        C = 2 * (B - (X / A ** 2))
        ans = degrees(atan(C / A))
    print(ans)


if __name__ == "__main__":
    main()