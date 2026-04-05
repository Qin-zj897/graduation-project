def solve(h, n):
    i=1
    j=h
    while i<=n-1:
        h=h/2
        j+=2*h
        i+=1
    return '%.2f'%j


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
