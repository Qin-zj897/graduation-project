def solve(x):
    a=x.count(max(x))
    b=x,count(min(x))
    if len(x)>0:
        for i in range(a):
            del x[x.index(max(x))]
    if len(x)>0:
        for i in range(b):
            del x[x.index(min(x))]
    return x


if __name__ == '__main__':
    x = eval(input())
    result = solve(x)
    print(result)
