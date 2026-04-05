def solve(a):
    b = a.copy()
    m=max(a)
    n=min(a)
    for i in b:
        if i == m or i== n:
            a.remove(i)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
