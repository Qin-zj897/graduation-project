# -*- coding: utf-8 -*-
# 统计代码数量

import os
import json

def analyze_student_codes():
    stats = {}
    
    for problem in ['2910', '3039']:
        stats[problem] = {
            'success_count': 0,
            'failure_count': 0,
            'total': 0
        }
        
        success_path = os.path.join(problem, 'success')
        failure_path = os.path.join(problem, 'failure')
        
        # 统计数量
        if os.path.exists(success_path):
            success_files = [f for f in os.listdir(success_path) if f.endswith('.py')]
            stats[problem]['success_count'] = len(success_files)
            
        if os.path.exists(failure_path):
            failure_files = [f for f in os.listdir(failure_path) if f.endswith('.py')]
            stats[problem]['failure_count'] = len(failure_files)
            
        stats[problem]['total'] = stats[problem]['success_count'] + stats[problem]['failure_count']
    
    return stats

# 运行
stats = analyze_student_codes()
print(json.dumps(stats, indent=2, ensure_ascii=False))