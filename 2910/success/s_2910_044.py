def solve(a, b):
    c=a
    for i in range(b-1):
        x=(0.5)**i
        d=a*x
        c+=d
    return "%.2f"%c


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
