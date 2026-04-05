def solve(h, N):
    H=h
    for i in range(N-1):
        a=h*0.5**(i+1)
        H+=a*2
    return "%.2f"%(H)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
