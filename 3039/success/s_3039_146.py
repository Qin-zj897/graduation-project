def solve(a):
    b=max(a);c=min(a)
    for i in d:
        if i==b:
            a.remove(i)
        elif i==c:
            a.remove(i)
        return a


if __name__ == '__main__':
    a = eval(input());d=a.copy()
    result = solve(a)
    print(result)
