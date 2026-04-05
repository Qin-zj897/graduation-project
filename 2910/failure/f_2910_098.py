def solve(a, b):
    for x in range(b-1):
        sum=a
        sum+=a*(0.5)**(x)
    return '%.2f'%(sum)


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    result = solve(a, b)
    print(result)
