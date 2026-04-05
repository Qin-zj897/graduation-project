def solve(h, N):
    s=0
    while N-1>0:
        s=s+h*0.5
        h=h*0.5
        N=N-1
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
