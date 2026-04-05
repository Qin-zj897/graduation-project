def solve(ls1):
    a = max(ls1)
    b = min(ls1)
    if a == b:
        return "[]"
    else:
        while a in ls1:
            ls1.remove(a)
        while b in ls1:
            ls1.remove(b)
            return ls1


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
