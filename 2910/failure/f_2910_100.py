def solve(h, N):
    if N == 1:
        return "%.2f"%h
    else:
        s=h+2*(h/2)**N-1
        return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
