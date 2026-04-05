def solve(h, N):
    H=h
    if N==1:
        return "%.2f"%(H)
    else:
        for i in range(N):
            H=H+2*h*0.5**i
        return "%.2f"%(H)


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
