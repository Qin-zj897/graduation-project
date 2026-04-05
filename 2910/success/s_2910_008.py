def solve(h, N):
    l=2*h*(1-(1/2)**N)+h*(1-(1/2)**(N-1))
    return "%.2f"%l


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
