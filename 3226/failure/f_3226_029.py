def solve(nums):
    def search(y):
        for x in y:
            if y.count(x)>len(y)//2:
                return x
            else
            return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
