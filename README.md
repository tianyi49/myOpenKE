### original_test.py
原来的开源框架的测试代码
### struct_des_test.py
用于知识补全实验的模型的代码

### genKGCData.py
用于生成知识图谱补全实验的数据，主要是加载模型，预测新的CWE与CWE之间可能导致CWEChain的关系和cwe新的相关cve：
- 1.根据get_reason_data得到的测试数据加载模型进行预测前10的尾实体
- 2.排除已有的CWEChain关系和cwe相关cve
- 3 计算平均三元组距离分数，设定阈值，得到最终补全的CWEChain关系和cwe相关cve

