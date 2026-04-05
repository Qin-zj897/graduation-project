def solve(a, N):
    H=a
    if N==1:
        return "%.2f"%a
    else:
        for i in range(1,N):
            H=H+2*a*0.5**i
        return "%.2f"%H


if __name__ == '__main__':
    a = eval(input())
    N = eval(input())
    result = solve(a, N)
    print(result)
