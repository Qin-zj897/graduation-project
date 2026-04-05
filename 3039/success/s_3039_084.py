def solve(ls):
    a=ls.count(max(ls))
    b=ls.count(min(ls))
    i=0
    while i<a:
        ls.remove(max(ls))
        i+=1
    if len(ls)>=1:
        j=0
        while j<b:
            ls.remove(min(ls))
            j+=1
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
