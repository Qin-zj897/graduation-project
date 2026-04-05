def solve(nums):
    def search(nus):
        for i in range (len(nus)):
            if nus.count(i)>((len(nus))/2):
                return i
            else:
                continue
        return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
