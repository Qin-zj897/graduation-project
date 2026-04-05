def solve(x):
    M=max(x)
    m=min(x)
    while M in x:
        x.remove(M)
    while m in x:
        x.remove(m)
    return x


if __name__ == '__main__':
    x = eval(input())
    result = solve(x)
    print(result)
