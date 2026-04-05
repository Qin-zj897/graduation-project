def solve(h, n):
    total=h 
    for x in range(n-1):
        total+=total*(0.5**x)
    return format(total,'.2f')


if __name__ == '__main__':
    h = eval(input())
    n = eval(input()) 
    result = solve(h, n)
    print(result)
