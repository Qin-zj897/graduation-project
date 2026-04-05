def solve(t):
    t.sort()
    min=t[0]
    max=t[-1]
    for i in t:
        if i==min:
            t.remove(i)
        elif i==max:
            t.remove(i)
    return t


if __name__ == '__main__':
    t = eval(input())
    result = solve(t)
    print(result)
