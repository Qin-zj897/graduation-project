def solve(a, b):
    c=a*4-4*a*0.5**b-a
    return "%.2f"%c


if __name__ == '__main__':
    a = eval(input())
    b = eval(input())
    result = solve(a, b)
    print(result)
