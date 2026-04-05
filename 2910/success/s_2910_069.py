def solve(h, N):
    res=0
    while N>1:
        res+=(h*3/2)
        h=h/2
        N-=1
    res+=h
    return "%.2f"%res


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
