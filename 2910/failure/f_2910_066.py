def solve(h, n):
    s=h
    for i in range(n):
        s+=h*0.5**i
    return '%.2f'%s


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
