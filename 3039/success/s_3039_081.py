def solve(List):
    from re import M


    List1 = List.copy()
    for i in List:
        Max = max(List)
        Min = min(List)
        if Max == max(List1):
            List1.remove(Max)
        elif Min == min(List1):
            List1.remove(Min)
        else:
            break
    return List1


if __name__ == '__main__':
    List = eval(input())
    result = solve(List)
    print(result)
