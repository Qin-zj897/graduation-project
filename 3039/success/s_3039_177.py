def solve(input_list):
    max_value = max(input_list)
    min_value = min(input_list)
    output_list = [num for num in input_list if num != max_value and num != min_value]
    return output_list


if __name__ == '__main__':
    input_list = eval(input().strip())
    result = solve(input_list)
    print(result)
