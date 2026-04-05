def solve(h, n):
    a = h
    for x in range(n-1):
        a = a+h
        h = h*0.5
    return "%.2f"%a


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
