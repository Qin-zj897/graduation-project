# -*- coding: utf-8 -*-
# xml文件解析
import xml.etree.ElementTree as ET

def parse_testcases(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    testcases = []
    
    for elem in root:
        if elem.tag.startswith('testData'):
            input_elem = elem.find('input')
            output_elem = elem.find('output')
            
            if input_elem is not None and output_elem is not None:
                input_str = input_elem.text.strip() if input_elem.text else ''
                output_str = output_elem.text.strip() if output_elem.text else ''
                
                testcases.append({
                    'input': input_str,
                    'expected': output_str
                })
    
    return testcases

# 测试
testcases_2910 = parse_testcases('2910.xml')
testcases_3039 = parse_testcases('3039.xml')

print("2910题有", len(testcases_2910), "个测试用例")
print("3039题有", len(testcases_3039), "个测试用例")