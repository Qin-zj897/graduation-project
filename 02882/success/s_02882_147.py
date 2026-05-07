def main():
    from math import atan, degrees
    a, b, x = (int(i) for i in input().split())
    if x/a >= a*b/2:
        tmp = (2*b - ((2*x)/(a*a))) / a
        print(degrees(atan(tmp)))
    else:
        tmp = b/((x/a)*(2/b))
        k = atan(tmp)
        print(degrees(k))


if __name__ == '__main__':
    main()
