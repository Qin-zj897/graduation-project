def solve(h, N):
    sum=h
    for i in range(N-1):
        sum+=2*h/2
        h=h/2
    return "%.2f"%(sum)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
