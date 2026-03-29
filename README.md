# graduation-project
毕业设计

20260323-20260329
初步设计了测试用例生成模块，为了更好阅读，方法中的参数都使用了类型注解<img width="1164" height="816" alt="8e268b2a77f7959be5d7b275b84073eb" src="https://github.com/user-attachments/assets/780a0287-c21f-4100-bc2f-8d4d0d48dea8" />
具体功能在testcase_generator_doc.md中查看

可以使用test_testcase.py进行测试，修改默认读取的文件或者在命令行添加参数都可以，输出的结果是精简后的测试用例集和覆盖率统计。
关于覆盖率：测试的时候发现有些边、块是一直不可达的虚拟边，就把它们剔除了
修改了静态分析工具，在get_branch_constraint_map()方法的返回字典中，新增了'lineno'字段，让TestcaseGenerator能通过静态分析直接获取每个分支的行号
