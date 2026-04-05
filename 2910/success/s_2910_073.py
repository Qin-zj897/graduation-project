def solve(a, N):
    h=a
    for x in range(N-1):
        a=a*0.5
        h+=a*2
    return "%.2f"%(h)


if __name__ == '__main__':
    a = eval(input())
    N = eval(input())
    result = solve(a, N)
    print(result)
