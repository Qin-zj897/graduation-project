def solve(nums):
    def search(x):
        test=0
        for x1 in x:
            if x.count(x1)>len(x)/2:
                test=x1
        if test !=0:
            return test
        elif test==0:
            return"False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
