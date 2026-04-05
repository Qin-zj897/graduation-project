def solve(nums):
    def search(x):
        for i in x:
            while x.count(i)>len(x)//2:
                return i
            else: 
                return "False"






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
