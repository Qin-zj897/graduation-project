def solve(h, N):
    L = h
    if N==1:
        return "%.2f"%h
    else:
        for i in range(1,N):
            L = L+2*h*0.5**i
        return "%.2f"%L


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
