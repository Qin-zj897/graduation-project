def solve(a):
    b=max(a)
    c=min(a)
    d=a.copy()
    for i in a:
    	if i==b or i==c:
    		d.remove(i)
    return d


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
