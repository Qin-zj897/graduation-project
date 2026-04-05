def solve(ls):
    b=max(ls)
    c=min(ls)
    nums=ls.copy()
    for i in nums:
        if i==b or i==c:
            ls.remove(i)   
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
