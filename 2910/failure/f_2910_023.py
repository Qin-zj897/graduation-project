def solve(h, N):
    s=h
    if N>1:
        for i in range(1,N):
            s=h+2*(h/2**N)
            return f'{s:.2f}'
    else:
        return f'{s:.2f}'


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
