def solve(ls):
    x=max(ls)
    y=min(ls)
    ls1=[]
    for i in range(len(ls)):
        if ls[i]!=x and ls[i]!=y:
            ls1.append(ls[i])
    return ls1


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
