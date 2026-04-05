def solve(h, n):
    s = h
    if n > 1:
        for x in range(n-1):
            h = h/2
            s += h*2
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
