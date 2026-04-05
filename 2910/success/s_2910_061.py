def solve(h, c):
    h=4*h*(1-(1/2)**c)-h
    return "%.2f"%h


if __name__ == '__main__':
    h = eval(input())
    c = eval(input())
    result = solve(h, c)
    print(result)
