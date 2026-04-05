def solve(h, n):
    H=h
    if n==1:
        return "{:.2f}".format(h)
    else:
        for i in range(1,n):
            H=H+2*h*0.5**i
        return "{:.2f}".format(H)


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
