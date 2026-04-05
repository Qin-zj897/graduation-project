def solve(nums):
    def search(nums):
        n1=[]
        for x in nums:
            n1.append(nums.count(x))
            n2=max(n1)    
        if n2>len(nums)/2:
           return max(nums,key=nums.count)  
        else:
            return  "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
