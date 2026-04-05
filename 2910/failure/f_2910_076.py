def solve(h, N):
    s=h+h*(1-0.5**(N-1))/0.25
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
