def solve(input_list):
    def remove_extremes(lst):
        min_val = min(lst)
        max_val = max(lst)
        new_lst = [x for x in lst if x != min_val and x != max_val]
        return new_lst


    lst = eval(input_list)
    new_lst = remove_extremes(lst)
    return new_lst


if __name__ == '__main__':
    input_list =  input()
    result = solve(input_list)
    print(result)
