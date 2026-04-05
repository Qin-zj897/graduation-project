def solve(nums):
    def search(n):
        a = n.count(max(n,key = n.count))
        b = max(n,key = n.count)
        if a > len(n)//2:
            return b
        else:
            return 'False'





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
