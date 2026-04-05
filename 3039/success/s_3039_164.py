def solve(n):
    b=max(n)
    c=min(n)
    while b in n:
        n.remove(b)
    while c in n:
        n.remove(c)
    return n


if __name__ == '__main__':
    n = eval(input())
    result = solve(n)
    print(result)
