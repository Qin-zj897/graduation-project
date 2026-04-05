def solve(h, N):
    from cgi import print_environ


    s=0
    for x in range(1,N+1):
              H=h/2
              s=s+H*2
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
