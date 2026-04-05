def solve(ls):
    ls2 = ls*n
    ls3 = [x*x for x in ls2]
    ls4=[]
    for x in ls3:
        if x not in ls4:
            ls4.append(x)
    return ls4
if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
