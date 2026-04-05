def solve(a):
    b=max(a)
    c=min(a)
    for i in a:
    	if x==b or x==c:
    		a.remove(i)
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
