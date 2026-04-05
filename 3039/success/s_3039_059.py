def solve(ls1):
    ls2 = []
    for x in ls1:
        if x > min(ls1) and x < max(ls1):
            ls2.append(x)
    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
