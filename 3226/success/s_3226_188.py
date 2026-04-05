def solve(nums):
    def search(num):
        n=len(num)//2
        lit=[]
        for i in range(len(num)):
            a=num.count(num[i])
            if a>n:
                return num[i]
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
