# graduation-project
毕业设计

20260119-20260125
重写了静态分析工具类static_analyzer.py和一个简陋的测试代码test.py。可以实现1.提取常量值。2.构建CFG。3.谓词挖掘。4.简单的后向切片。5.输入结构推断。6.推断变量类型。7.数据依赖分析。
测试代码使用1.
def search(nums):
    for x in nums:
        if nums.count(x) > len(nums) // 2:
            return x
    else:
        return False

nums = eval(input())
y = search(nums)
print(y)
2.
h=eval(input())
n=eval(input())
alp=0.5
sum1=h
for i in range(1,n):
    sum1+=h*alp*2
    h=h*alp
print(\"%.2f\" %sum1)
3.
a = eval(input())
b = max(a)
c = min(a)
i = 0
while i < len(a):
    if b == a[i]:
        a.remove(b)
        i = i - 1
    elif c == a[i]:
        a.remove(c)
        i = i - 1
    i = i + 1
print(a)
三道题的参考代码，问题有：变量类型推断没有完全实现，存在未知类型；切片作用不明


修改了代码，统一了每个部分的输出格式，可以正常提取链式比较的左右边界，def新增了推断解包赋值的类型，处理带注解的赋值，处理增强赋值等；输入推断可以支持多种输入模式
仍存在的问题：变量类型推断未实现，eval(input())推断出Any后没有根据上下文推断出具体类型；CFG部分代码修改后效果未验证
