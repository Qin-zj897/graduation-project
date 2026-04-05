def solve(list1):
    list1=list(list1)
    a=max(list1)
    b=min(list1)
    jishu1=0
    jishu2=0
    for x in list1:
        if x==a or x==b:
            list1[jishu2]='c'
            jishu1+=1
        jishu2+=1
    while jishu1>0:
        list1.remove('c')
        jishu1-=1
    return list1


if __name__ == '__main__':
    list1 = eval(input())
    result = solve(list1)
    print(result)
