def solve(n):
    a=max(n)
    b=min(n)
    while a in n:
        n.remove(a)
    while b in n:
        n.remove(b)
    return n


if __name__ == '__main__':
    n = eval(input())
    result = solve(n)
    print(result)
