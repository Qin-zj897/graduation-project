def solve(h, n):
    tot=h
    for i in range(n-1):
        tot+=h
        h=h/2
    return "%.2f"%tot


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
