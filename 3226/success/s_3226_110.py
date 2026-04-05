def solve(nums):
    def search(nums):
        x=0
        b=[]
        for i in range (len(nums)):
            a=nums.count(nums[i])
            if a >len(nums)/2:
                x+=1
                b.append(nums[i])    
            else:
                pass
        if not x==0:
            return b[0]     
        else: 
            return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
