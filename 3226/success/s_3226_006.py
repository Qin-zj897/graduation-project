def solve(nums):
    def search(l):
        l=list(l)
        for x in l:
            if l.count(x)>len(l)/2:
                return x
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
