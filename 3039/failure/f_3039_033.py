def solve(l):
    a = max(l)
    b = min(l)
    for i in l:
        if i == a:
            l.remove(i)
    for i in l:
        if i == b:
            l.remove(i)        
    return l


if __name__ == '__main__':
    l = eval(input())
    result = solve(l)
    print(result)
