def solve(nums):
    def search(list):
        n = len(list)
        for i in list:
            if list.count(i) > n/2:
                return i

            return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
