# -*- coding: utf-8 -*-
import math
def main():
    a, b, x=map(int, input().split())  #複数数値入力　「A B」みたいなスペース空いた入力のとき
    rev = 0
    if x > (a*a*b/2):
        #print("lar")
        x = a*a*b - x
        print(90-math.degrees(math.atan(a*a*a/2/x)))
    else:
        print(math.degrees(math.atan(b*b*a/2/x)))

if __name__ == '__main__':
    main()