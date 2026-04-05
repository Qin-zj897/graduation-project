def solve(h, n):
    eva=h
    if n==1:
        return "%.2f"%h
    else:
        while n>0:
            eva+=h*0.5*2
            h*=0.5
            n-=1
        return "%0.2f"%eva


if __name__ == '__main__':
    h = eval(input())
    n = eval(input())-1
    result = solve(h, n)
    print(result)
