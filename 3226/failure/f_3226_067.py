def solve(nums):
    def search(n):
        for i in n:
            if n.count(i) >len(n)//2:
                return i
        else:
              return('Flase')





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
