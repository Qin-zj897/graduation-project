def solve(h, N):
    s=h*(1-0.5**(N+2))/0.5
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
