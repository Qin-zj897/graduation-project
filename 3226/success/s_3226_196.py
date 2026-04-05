def solve(nums):
    def search(nums):
        m=[]
        for i in nums:
            if nums.count(i)>len(nums)//2:
                m.append(i)
                return i
            else:
                continue 
        if m==[]:
            return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
