def solve(h, N):
    sum1=h
    for i in range(N-1):
        sum1=h+h/2
    return "%.2f"%sum1


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
