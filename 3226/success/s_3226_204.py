def solve(nums):
    def search(ls):
        l=len(ls)
        for i in ls:
            c=ls.count(i)
            if c>(l/2):
                return(i)
            else:
                return "False"







    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
