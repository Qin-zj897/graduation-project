def solve(a, b):
    h=10
    a=20
    for x in range(b-1):
        a/=2
        h+=a
    return "%.2f"%(h)


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
