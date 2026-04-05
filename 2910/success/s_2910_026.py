def solve(a, b):
    s=a
    for x in range(b-1):
        c=a/2
        s=s+c*2
        a=c
    return '%.2f'%s


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
