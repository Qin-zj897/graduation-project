def solve(h, n):
    sum=h
    for i in range(n-1):
        sum+=h
        h=h*0.5
    return "%.2f"%sum


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
