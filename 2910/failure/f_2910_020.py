def solve(h, n):
    s=h
    for i in range(0,n):
        s+=h*(0.5**i)
    return s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
