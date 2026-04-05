def solve(nums):
    def search(a):
        for i in a:
            if a.count(i)>len(a)//2:
                return i
        else:
             return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
