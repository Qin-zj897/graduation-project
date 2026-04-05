def solve(nums):
    def search(n):
        for x in n:
            s=0
            for y in n:
                if x==y : s+=1
            if s>len(n)/2 : return x
        return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
