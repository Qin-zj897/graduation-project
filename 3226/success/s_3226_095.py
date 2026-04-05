def solve(nums):
    def search(a):
        for x in a:
            if a.count(x)>len(a)//2:
                return x
                break
            else:
                continue
        return 'False'









    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
