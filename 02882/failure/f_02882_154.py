from numpy import arctan, pi

def main():
    a, b, x = map(int, input().split())
    if a ** 2 * b > 2 * x:
        ans = arctan(2*x/(a*b*b))
    elif a ** 2 * b < 2 * x:
        ans = arctan(a*a*a/2/(a*a*b-x))
    else:
        ans = arctan(a/b)
    print(90 - ans*180/pi)


if __name__ == '__main__': main()