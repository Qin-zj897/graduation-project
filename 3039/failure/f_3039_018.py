def solve(a):
    b=max(a)
    c=min(a)
    a.remove(b)
    a.remove(c)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
