#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 原答案没有指出三位数的数量，添加无重复三位数的数量
import optimization
#s = [1,4,3,2,7,3,6,3,2,4,5,4] #这是一组题解。
#optimization.printschedule(s)
####################### 成本函数 寻找一组能使成本函数值最小的解############
# reload(optimization)
# print optimization.schedulecost(s)
####################### 随机解############

# reload(optimization)
# domain =[(0,9)*(len(optimization.people)*2)]
# s = optimization.randomoptimize(domain,optimization.schedulecost) # 会调用schedulecost()
# print optimization.schedulecost(s)
# optimization.printschedule(s)
reload(optimization)
domain = [(0,9)*(len(optimization.people)*2)]
s = optimization.randomoptimize(domain,optimization.schedulecost)
print optimization.schedulecost(s)
####