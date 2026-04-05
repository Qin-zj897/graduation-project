def solve(list):
    max=max(list)
    min=min(list)
    list2=[max,min]
    list3=[]
    for x in list:
        if x not in list2:
            list3.append(x)
    return list3


if __name__ == '__main__':
    list = eval(input())
    result = solve(list)
    print(result)
