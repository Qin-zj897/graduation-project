def solve(a):
    b=a.count(max(a))
    c=a.count(min(a))
    for x in range(b):
        a.remove(max(a))
    if a!=[]:
        for x in range(c):
            a.remove(min(a))
    else:
        pass
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
