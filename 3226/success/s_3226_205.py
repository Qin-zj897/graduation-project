def solve(nums):
    def search(n):
        l=len(n)
        x=round(l//2)    
        for i in n:
            c=n.count(i)
            if c>x:
               return i
            else:
                return "False"







    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
