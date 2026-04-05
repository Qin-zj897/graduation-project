def solve(l):
    a = max(l)
    b = min(l)
    l2=[]
    for i in l:
        if i == a:
            continue
        else:
            l2.append(i)
    for i in l:
        if i == b:
            continue
        else:
            l2.append(i)        
    return l


if __name__ == '__main__':
    l = eval(input())
    result = solve(l)
    print(result)
