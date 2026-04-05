def solve(h, N):
    height=h
    for i in range(1,N):
        h=h/2
        height+=h+2
    return "%.2f"%(height)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
