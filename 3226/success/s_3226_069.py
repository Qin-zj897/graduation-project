def solve(nums):
    def search(n):
        for i in n:
            a=n.count(i)
            if a>len(n)//2:
                return(i)
            else:
                return 'False'





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
