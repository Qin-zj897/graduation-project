def solve(a, b):
    c=0
    for i in range(b):
        c+=a*0.5**i
    return "%.2f"%c


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
