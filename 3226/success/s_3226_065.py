def solve(nums):
    def search(n): 
        for i in n:
            i_count = n.count(i)
            if i_count>len(n)//2:
                return i
            else:
                return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
