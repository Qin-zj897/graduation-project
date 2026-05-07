import math

def main():
    a,b,x = list(map(int,input().split()))

    v = a**2*b
    ans = 0
    if x <= v/2:
        ans = math.degrees(math.atan(a*b**2/(2*x)))
    else:
        ans = math.degrees(math.atan(2*(a**2*b-x)/(a**3)))
    print(ans)

if __name__ == '__main__':
    main()
