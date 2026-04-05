def solve(h, n):
    if n ==1:
        l = h
    elif n ==2:
        l =2*h
    else:
        l =2*h
        for i in range(1,n-1):
            h = h*0.5
            l = l+h
    return '%.2f'%l


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
