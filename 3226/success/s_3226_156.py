def solve(nums):
    def search(a):
        b = []
        for x in a:
            if a.count(x)>len(a)//2:
                b.append(x)
            if b==[]:return False
            else:
                 return  max(b)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
