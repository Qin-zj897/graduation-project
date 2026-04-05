def solve(h, N):
    m=h
    for i in range(N-1):
        m+=h*(0.5)**(i)
    return "%.2f"%m


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
