def solve(a):
    m=max(a)
    n=min(a)
    list1=a.copy()
    for x in a:
        if x==m or x==n:
            list1.remove(x)
    return list1


if __name__ == '__main__':
    a = list(eval(input()))
    result = solve(a)
    print(result)
