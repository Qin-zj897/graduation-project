def solve(m):
    ma=max(m)
    mi=min(m)
    lm=m[:]
    for i in lm:
        if i==ma:
            m.remove(i)
        elif i==mi:
            m.remove(i)
    return m


if __name__ == '__main__':
    m = eval(input())
    result = solve(m)
    print(result)
