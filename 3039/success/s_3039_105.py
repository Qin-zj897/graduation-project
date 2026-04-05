def solve(c):
    z=min(c)
    b=max(c)
    return [x for x in c if x!=z and x!=b]


if __name__ == '__main__':
    c = eval(input())
    result = solve(c)
    print(result)
