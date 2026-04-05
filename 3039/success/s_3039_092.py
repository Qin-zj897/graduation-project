def solve(k):
    i=max(k)
    p=min(k)
    m=k.count(i)
    n=k.count(p)
    if i==p:
            k.clear()
    else:
        for u in range(m):
                k.remove(i)
        for b in range(n):
                k.remove(p)        
    return k


if __name__ == '__main__':
    k = eval(input())
    result = solve(k)
    print(result)
