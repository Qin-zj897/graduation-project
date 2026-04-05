def solve(nums):
    def search(a):
        m=0
        for i in a:
            if a.count(i)>len(a)//2:
                m+=1
                return i
        if m ==0:
            return False          






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
