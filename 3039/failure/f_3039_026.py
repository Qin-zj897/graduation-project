def solve(a):
    b = max(a)    
    c = min(a)
    m = a.count(b)
    n = a.count(c)
    for i in range(m):
        a.remove(b)
    for i in range(n):
        a.remove(c)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
