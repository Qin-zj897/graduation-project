def solve(h, N):
    h1=0
    h2=0
    if N>1:
        for x in range(N):
            h1=h/2**x
            h2=h1*2+h2
        return "%.2f"%(h2-h)
    else:
        h2=10
        return "%.2f"%(h2)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
