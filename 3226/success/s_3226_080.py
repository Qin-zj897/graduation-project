def solve(nums):
    def search(list):
        b=0
        for x in list:
            n=list.count(x)
            if n>len(list)//2:
                a=x
                b+=1
            else:
                continue
        if b==0:
            return False
        else:
            return a





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
