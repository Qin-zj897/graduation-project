def solve(height, cishu):
    sums = height
    for x in range(cishu-1):
        sums+=height*(0.5)**x
    return "%.2f"%sums


if __name__ == '__main__':
    height = eval(input())
    cishu = eval(input())
    result = solve(height, cishu)
    print(result)
