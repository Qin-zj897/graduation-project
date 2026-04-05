def solve(a):
    b=max(a)
    c=min(a)
    if b in a:
        a.remove(b)
    if c in a:
        a.remove(c)     
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
