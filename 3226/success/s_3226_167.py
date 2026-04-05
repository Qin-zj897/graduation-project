def solve(nums):
    def search(x):
        for a in x:
            if x.count(a)>len(x)//2:
               return a 
        return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
