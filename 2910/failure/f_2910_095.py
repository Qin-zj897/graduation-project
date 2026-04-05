def solve(h, n):
    r=0
    while n>1:
        r+=h*3/2
        h=h/2
        n-=1
        r+=h
    return "%.2f"%r


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
