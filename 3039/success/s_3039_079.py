def solve(a):
    b=max(a)
    c=min(a)
    for i in range(len(a)-1,-1,-1):
        if a[i]==b:
              a.remove(a[i])
        elif a[i]==c:
              a.remove(a[i])
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
