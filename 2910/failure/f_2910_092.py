def solve(h, N):
    m=h*2
    i=0
    while i<N:
        h/=2
        m+=h
        i+=1
    return "%.2f"%m


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
