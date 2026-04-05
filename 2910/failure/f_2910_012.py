def solve(h, N):
    s=0
    for i in range(N+1):
        s+=h/2**i*2
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
