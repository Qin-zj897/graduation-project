def solve(h, N):
    sum=h
    for i in range(N):
        h=h*0.5
        sum=sum+2*h
    return "%.2f"%sum


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
