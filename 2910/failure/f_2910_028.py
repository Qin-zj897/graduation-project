def solve(h, N):
    s=h
    if N>1:
        for i in range(1,N):
            c=h*(1/2)**(i)
            s=h+2*c
        return f'{s:.2f}'
    else:
        return f'{s:.2f}'


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
