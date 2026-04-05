def solve(h, h):
    eva=h
    if n==1:
        return "%,2"%h
    else:
        while n>0:
            eva+=h*0.5*2
            h*=0.5
            n-=1
        return "%.2f"%eva


if __name__ == '__main__':
    h = eval(input())
    h = eval(input())-1
    result = solve(h, h)
    print(result)
