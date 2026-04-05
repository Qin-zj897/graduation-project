def solve(h, n):
    if n==1:
        l=h
    elif n==2:
        l=h*2
    else:
        l=h*2
        for i in range(1,n-1):
            h = h*0.5
            l+=h
    return "%.2f"%l


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
