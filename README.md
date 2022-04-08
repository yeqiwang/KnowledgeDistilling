# 知识蒸馏
该代码是基于tensorflow2的知识整理样例

This is a example of knowledge distilling based on tensorflow2

## 文件描述
代码包括一个main.py文件和一个distilling python包

其中，distilling包中分别包含：
- data_load.py：加载数据，获取dataset实例
- model_structure.py：定义模型结构，获取教师模型或学生模型
- distilling.py：知识蒸馏实现
- train_config.py：配置文件，包括数据配置、训练配置
- model_train.py：模型训练

## 运行步骤
在终端运行main.py
```python
> python main.py --model teacher
```

其中model参数包括teacher、student、distilling三种

首先，运行teacher、student，获取对应的训练weights（运行后保存在cheakpoint文件中）

然后，运行distilling，得到知识蒸馏下的学生模型

## 结果示例
运行30个epoch的情况下，

teacher **loss: 0.3015  acc: 0.8939**

student **loss: 0.3376  acc: 0.8810**

student(distilling) **loss: 0.3367  acc: 0.8816**

## 博客链接
博客代码与github中代码存在不同，以github中代码为准

https://blog.csdn.net/For_learning/article/details/117304450

