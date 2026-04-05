def solve(list):
    from ast import Del


    a=max(list)
    b=min(list)
    list1=[]
    for x in list:
        if x!=a and x!=b:
           list1.append(x)
    return list1


if __name__ == '__main__':
    list = eval(input())
    result = solve(list)
    print(result)
