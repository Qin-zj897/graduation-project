def solve(h, n):
    total=h 
    while n>0:
        for x in range(n):
            total=total+total*0.5**n
    return format(total,'.2f')


if __name__ == '__main__':
    h = eval(input())
    n = eval(input()) 
    result = solve(h, n)
    print(result)
