def solve(list1):
    a=max(list1)
    b=min(list1)
    cs1=0
    cs2=0
    for x in list1:
        if x==a or x==b:
            list1[cs2]="d"
            cs1+=1
        cs2+=1
    while cs1>0:
        list1.remove("d")
        cs1+=-1
    return list1


if __name__ == '__main__':
    list1 = eval(input())
    result = solve(list1)
    print(result)
