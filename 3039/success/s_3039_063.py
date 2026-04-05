def solve(n1):
    n2=max(n1)
    n3=min(n1)
    for x in range(len(n1)):        
        if n2 in n1:
         n1.remove(n2)
        if n3 in n1:
            n1.remove(n3)
    return n1


if __name__ == '__main__':
    n1 = eval(input())
    result = solve(n1)
    print(result)
