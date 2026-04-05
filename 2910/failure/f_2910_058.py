def solve(h, N):
    if N==1:
       s=h
    elif N>=2:
        for x in range(2,N+1):
            s=s+2*h*0.5**(x-1)
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
