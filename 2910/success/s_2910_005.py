def solve(h, N):
    s=h
    h1=h
    for i in range(N):
        h1=h1*0.5
        s+=h1*2
    s=s-h*(0.5**(N-1))
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
