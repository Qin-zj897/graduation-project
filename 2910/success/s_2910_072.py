def solve(h, n):
    x=h
    for i in range(n):
        if i==0:
            pass
        if i>0:
            x+=h*(0.5**(i-1))
    return "{:.2f}".format(x)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
