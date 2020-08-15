#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
import math

# The dorms, each of which has two available spaces
dorms=['Zeus','Athena','Hercules','Bacchus','Pluto']

# People, along with their first and second choices
prefs=[('Toby', ('Bacchus', 'Hercules')),
       ('Steve', ('Zeus', 'Pluto')),
       ('Karen', ('Athena', 'Zeus')),
       ('Sarah', ('Zeus', 'Pluto')),
       ('Dave', ('Athena', 'Bacchus')), 
       ('Jeff', ('Hercules', 'Pluto')), 
       ('Fred', ('Pluto', 'Athena')), 
       ('Suzie', ('Bacchus', 'Hercules')), 
       ('Laura', ('Bacchus', 'Hercules')), 
       ('James', ('Hercules', 'Athena'))]

# [(0,9),(0,8),(0,7),(0,6),...,(0,0)]
domain=[(0,(len(dorms)*2)-i-1) for i in range(0,len(dorms)*2)]

def printsolution(vec):
  slots=[]
  # Create two slots for each dorm 每个宿舍建立2个槽 <type 'list'>: ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']
  for i in range(len(dorms)):slots+=[i,i] #<type 'list'>: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

  # Loop over each students assignment 遍历每个学士的安置情况
  for i in range(len(vec)):
    x=int(vec[i])

    # Choose the slot from the remaining ones 从剩余的槽中选择
    dorm=dorms[slots[x]]
    # Show the student and assigned dorm 输出学士及住的情况
    print(prefs[i][0],dorm)
    # Remove this slot 删除改槽
    del slots[x]

def dormcost(vec):
  cost=0
  # Create list a of slots 建立一个 槽序列
  slots=[0,0,1,1,2,2,3,3,4,4]

  # Loop over each student 遍历每个学生
  for i in range(len(vec)):
    x=int(vec[i])
    dorm=dorms[slots[x]]
    pref=prefs[i][1]
    # First choice costs 0, second choice costs 1 首选成本 0 次选 为 1
    if pref[0]==dorm: cost+=0
    elif pref[1]==dorm: cost+=1
    else: cost+=3
    # Not on the list costs 3 不在选择 列 为 3

    # Remove selected slot 删除选择槽
    del slots[x]
    
  return cost
def dormcosts(vec):
    cost = 0
    slots=[0,0]

    for i in range(len(vec)):
        x = int(vec[i])
        dorm=dorms[slots[x]]