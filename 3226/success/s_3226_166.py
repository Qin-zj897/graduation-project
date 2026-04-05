def solve(nums):
    def search(num):
        ns=[]
        for x in num:
            n=len(num)//2
            if num.count(x)>n and x not in ns:
                ns.append(x)
        if len(ns)>=1:
            return int(''.join(map(str,ns)))
        else: return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
