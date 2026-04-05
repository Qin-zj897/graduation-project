def solve(list):
    list2=[]
    a=max(list)
    b=min(list)
    for x in list:
        if x!=a and x!=b:
            list2.append(x)
    return list2


if __name__ == '__main__':
    list = eval(input())
    result = solve(list)
    print(result)
