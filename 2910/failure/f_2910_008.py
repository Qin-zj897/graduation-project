def solve(h, N):
    s=h
    for i in range(N+1):
        s+=2**i*(h/2)**i
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
