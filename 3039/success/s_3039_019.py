def solve(a):
    amax=max(a)
    amin=min(a)
    b=a.copy()
    for n in a:
        if n==amax or n==amin:
            b.remove(n)
    return b


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
