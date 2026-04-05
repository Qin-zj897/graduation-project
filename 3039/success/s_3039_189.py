def solve(input_list):
    # 读取输入的整数列表

    # 找到最大元素和最小元素
    max_num = max(input_list)
    min_num = min(input_list)

    # 删除最大元素和最小元素
    result_list = [num for num in input_list if num != max_num and num != min_num]

    # 输出删除后的列表
    return result_list


if __name__ == '__main__':
    input_list = list(map(int, input().strip('[]').split(',')))
    result = solve(input_list)
    print(result)
