def solve(a):
    b=max(a)
    c=min(a)
    a1=a.copy()
    for x in a:
          if x==c or x==b :
            a1.remove(x)
    return a1


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
