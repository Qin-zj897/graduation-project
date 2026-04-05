def solve(h, n):
    s=h+h*(1-0.5**(n-1))*2
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
