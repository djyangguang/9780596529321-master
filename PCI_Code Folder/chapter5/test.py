#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 原答案没有指出三位数的数量，添加无重复三位数的数量
import optimization
import dorm
#s = [1,4,3,2,7,3,6,3,2,4,5,4] #这是一组题解。
#optimization.printschedule(s)
####################### 成本函数 寻找一组能使成本函数值最小的解############
reload(optimization)
#print optimization.schedulecost(s)
####################### 随机解############

# reload(optimization)
# domain =[(0,9)*(len(optimization.people)*2)]
# s = optimization.randomoptimize(domain,optimization.schedulecost) # 会调用schedulecost()
# print optimization.schedulecost(s)
# optimization.printschedule(s)
#reload(optimization)
#domain = [(0,9)]*(len(optimization.people)*2)
#print domain
#s = optimization.randomoptimize(domain,optimization.schedulecost)
#print optimization.schedulecost(s)
#optimization.printschedule(s)
####### 爬坡法
#s = optimization.hillclimb(domain,optimization.schedulecost)
#print optimization.schedulecost(s)
#optimization.printschedule(s)
#################################退火

# s = optimization.annealingoptimize(domain,optimization.schedulecost)
# print optimization.schedulecost(s)
# optimization.printschedule(s)
#################遗传

# = optimization.geneticoptimize(domain,optimization.schedulecost)

#optimization.printschedule(s)
#####################优化】
reload(dorm)
s = optimization.randomoptimize(dorm.domain,dorm.dormcost) # 随机
#dorm.printsolution([0,0,0,0,0,0,0,0,0,0])
dorm.dormcost(s)
#optimization.geneticoptimize(dorm.domain,dorm.dormcost) # 遗传
dorm.printsolution(s)
#import  dorm
#dorm.printsolution([0,8,0,0,0,0,0,0,0,0])