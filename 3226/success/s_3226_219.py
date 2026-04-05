def solve(nums):
    def search(a):
        for i in a:
            b=a.count(i)
            if b>(len(a)/2):
                return i   
            else:
                return False
                break





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
