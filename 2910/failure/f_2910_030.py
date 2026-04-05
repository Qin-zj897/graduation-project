def solve(h, N):
    if N>1:
        for i in range(1,N):
            c=h*(1/2)**(i)
            d=h+2*c
        return f'{d:.2f}'
    else:
        return f'{h:.2f}'


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
