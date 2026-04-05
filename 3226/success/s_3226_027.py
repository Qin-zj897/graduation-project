def solve(nums):
    def search(a):
        b=len(a)//2
        for x in a:
            if a.count(x)>b:
                return x
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
