def solve(a):
    a=list(a)
    b=max(a)
    c=min(a)
    list2=[]

    for x in a:
        if x!=b and x!=c:
            list2.append(x)

    return list2


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
