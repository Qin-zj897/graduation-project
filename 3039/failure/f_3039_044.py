def solve(a):
    for i in a:
       if i >= max(a) or i <= min(a):
          a.remove(i)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
