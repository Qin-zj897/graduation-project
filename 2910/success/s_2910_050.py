def solve(h, n):
    t=2
    x=0
    s=h
    for i in range(n-1):
        s=s+(h/(t**x))
        x=x+1
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
