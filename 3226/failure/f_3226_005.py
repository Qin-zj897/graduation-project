def solve(nums):
    def search(l):
        l=list(l)
        i=[x for x in l if l.count(x)>len(l)/2]
        for x in i:
            while i.count(x)>1:
                i.remove(x)
        if len(i)==0:
            return False
        return i





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
