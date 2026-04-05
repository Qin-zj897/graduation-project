def solve(h, n):
    H=h
    for i in range(n-1):
        H+=h*0.5**(i)
    return "%.2f"%H


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
