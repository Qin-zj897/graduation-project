def solve(h, n):
    m=n
    if n==1:
        h=h
    else:
      while m>0:
        h=h+h*2*0.5**(m-1)
        m=m-2
    return "%.2f"%h


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())
    result = solve(h, n)
    print(result)
