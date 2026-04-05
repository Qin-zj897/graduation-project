def solve(a):
    max=max(a)
    min=min(a)
    for x in reversed(a):
        if x==max or x==min:
            a.remove(x)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
