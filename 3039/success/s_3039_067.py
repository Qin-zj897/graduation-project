def solve(list):
    a=max(list)
    b=min(list)
    list2=[]
    for x in list:
        if x==a:
            continue
        else:
            if x==b:
                continue
            else:
                list2.append(x)
    return list2


if __name__ == '__main__':
    list = eval(input())
    result = solve(list)
    print(result)
