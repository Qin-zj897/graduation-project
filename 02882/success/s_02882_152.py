#!/usr/bin/env python3
import math


def main():
    a, b, x = input_line(int, int, int)
    if a ** 2 * b / 2 < x:
        tt = 2 * (a ** 2 * b - x) / a ** 3
    else:
        tt = a * b ** 2 / (2 * x)
    t = math.degrees(math.atan(tt))
    print(t)


def input_line(*types):
    if len(types) == 1:
        return types[0](input())
    else:
        return [t(x) for t, x in zip(types, input().split())]


if __name__ == "__main__":
    main()
