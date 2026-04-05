def solve(shulie):
    max = max(shulie)
    min = min(shulie)
    xinshulie=[]
    for x in shulie:
        if x == max:
            pass
        elif x == min:
            pass
        else:
            xinshulie.append(x)
    return xinshulie


if __name__ == '__main__':
    shulie = eval(input())
    result = solve(shulie)
    print(result)
