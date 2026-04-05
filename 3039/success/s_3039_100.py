def solve(input_list):
    # 从键盘输入整数列表

    # 找到最大元素和最小元素的值
    max_value = max(input_list)
    min_value = min(input_list)

    # 删除最大元素和最小元素
    output_list = [num for num in input_list if num != max_value and num != min_value]

    # 输出结果
    return output_list


if __name__ == '__main__':
    input_list = eval(input().strip())
    result = solve(input_list)
    print(result)
