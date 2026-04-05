def solve(nums):
    def search(a):
        n=len(a)
        z=n//2
        for x in a:
            if a.count(x)>z:
                return x
            else:
             return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
