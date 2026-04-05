def solve(h, n):
    from tkinter import N


    if n==1:
        s=h
    else:
        s=h
        for i in range(n-1):
            s += h/(2**i)
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
