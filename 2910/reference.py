def solve(h, n):
    alp=0.5
    sum1=h
    for i in range(1,n):
        sum1+=h*alp*2
        h=h*alp
    return "%.2f" %sum1


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
