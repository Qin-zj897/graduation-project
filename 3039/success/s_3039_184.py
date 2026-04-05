def solve(a):
    b=max(a)
    c=min(a)
    d=[i for i in a if i!=b and i!=c]
    return d


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
