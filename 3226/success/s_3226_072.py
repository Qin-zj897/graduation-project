def solve(nums):
    def search(a):
        x=0
        for i in a:
            if a.count(i) > len(a)//2:
                x=i
            else:
                x="False"
        return x





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
