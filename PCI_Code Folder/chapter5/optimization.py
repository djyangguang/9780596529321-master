#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import random
import math

people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
# Laguardia NEW York 的LGA机场
destination='LGA'

flights={}
# 
for line in file('schedule.txt'):
  origin,dest,depart,arrive,price=line.strip().split(',')
  flights.setdefault((origin,dest),[])

  # Add details to the list of possible flights
  flights[(origin,dest)].append((depart,arrive,int(price)))

def getminutes(t):#给定时间在一天的分钟数
  x=time.strptime(t,'%H:%M')
  return x[3]*60+x[4]

def printschedule(r):
  for d in range(len(r)/2):
    name=people[d][0]
    origin=people[d][1]
    out=flights[(origin,destination)][int(r[d])]
    ret=flights[(destination,origin)][int(r[d+1])]
    print ('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
                                                  out[0],out[1],out[2],
                                                  ret[0],ret[1],ret[2]))

def schedulecost(sol):
  totalprice=0
  latestarrival=0
  earliestdep=24*60

  for d in range(len(sol)/2):
    # Get the inbound and outbound flights
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    
    # Total price is the price of all outbound and return flights
    totalprice+=outbound[2]
    totalprice+=returnf[2]
    
    # Track the latest arrival and earliest departure
    #记录最晚到达和最早离开时间
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
  # Every person must wait at the airport until the latest person arrives.
  # They also must arrive at the same time and wait for their flights.
  #每个人必须在机场等待最后一个人 他们必须在相同时间到达，并等候他们的航班。
  totalwait=0  
  for d in range(len(sol)/2):
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep  

  # Does this solution require an extra day of car rental? That'll be $50!
  if latestarrival>earliestdep: totalprice+=50
  
  return totalprice+totalwait
          #随机搜索
#domain 指定每个变量的最大值和最小值。（0,9） 每个人重复2次
#costf ：成本函数 产生1000次猜想。
def randomoptimize(domain,costf):
  best=999999999
  bestr=None
  for i in range(0,1000):
    # Create a random solution 创建随随机解。
    r=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
    
    # Get the cost 得到成本 调用schedulecost
    cost=costf(r)
    
    # Compare it to the best one so far 与目前最有结进行比较
    if cost<best:
      best=cost
      bestr=r
  return r
##################################爬山法
def hillclimb(domain,costf):
  # Create a random solution 创建一个随机解
  sol=[random.randint(domain[i][0],domain[i][1])
      for i in range(len(domain))]
  # Main loop <type 'list'>: [3, 7, 4, 2, 9, 5, 0, 8, 2, 0, 1, 5]
  while 1:
    # Create list of neighboring solutions 创建相邻解的列表
    neighbors=[]
    
    for j in range(len(domain)):
      # One away in each direction 在每个方向上相对于原值偏离一点
      if sol[j]>domain[j][0]:
        neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
      if sol[j]<domain[j][1]:
        neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

    # See what the best solution amongst the neighbors is 在相邻中寻找最优解。
    current=costf(sol)
    best=current
    for j in range(len(neighbors)):
      cost=costf(neighbors[j])
      if cost<best:
        best=cost
        sol=neighbors[j]

    # If there's no improvement, then we've reached the top 如果没有最好的解
    if best==current:
      break
  return sol
#退火  domain ： <type 'list'>: [(0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 9)]
def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
  # Initialize the values randomly 随机初始化值 <type 'list'>: [2.0, 5.0, 6.0, 5.0, 4.0, 5.0, 4.0, 0.0, 5.0, 1.0, 0.0, 4.0]
  vec=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
  zz = 1
  while T>0.1:
    zz = zz +1
    # Choose one of the indices 选择一个索引值
    i=random.randint(0,len(domain)-1)
    #print zz+ '=======>'+ T
    print '%10s%0s%10s%30s ' % ('循环次数==',zz,'温度', T)
    # Choose a direction to change it 选择一个索引值改变的方向
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed 创建一个 解题列表，并改变其中一个值
    #<type 'list'>: [2.0, 5.0, 6.0, 5.0, 4.0, 5.0, 4.0, 0.0, 5.0, 1.0, 0.0, 4.0]
    vecb=vec[:]
    vecb[i]+=dir#<type 'list'>: [1.0, 5.0, 6.0, 5.0, 4.0, 5.0, 4.0, 0.0, 5.0, 1.0, 0.0, 4.0]
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current cost and the new cost 计算当前成本和新成本
    ea=costf(vec)# 4069
    eb=costf(vecb)#4091
    p=pow(math.e,(-eb-ea)/T)#0.44

    # Is it better, or does it make the probability 是否是最优解
    # cutoff?
    if (eb<ea or random.random()<p):
      vec=vecb      

    # Decrease the temperature 降低温度
    T=T*cool
  return vec

def geneticoptimize(domain,costf,popsize=50,step=1,
                    mutprob=0.2,elite=0.2,maxiter=100):
  # Mutation Operation 变异操作
  def mutate(vec):
    i=random.randint(0,len(domain)-1)
    if random.random()<0.5 and vec[i]>domain[i][0]:
      return vec[0:i]+[vec[i]-step]+vec[i+1:] 
    elif vec[i]<domain[i][1]:
      return vec[0:i]+[vec[i]+step]+vec[i+1:]
  
  # Crossover Operation 交叉操作
  def crossover(r1,r2):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

  # Build the initial population 构建初始总群
  pop=[]
  for i in range(popsize):
    vec=[random.randint(domain[i][0],domain[i][1]) 
         for i in range(len(domain))]
    pop.append(vec)
  
  # How many winners from each generation? 每一代有多少胜出
  topelite=int(elite*popsize)
  
  # Main loop 
  for i in range(maxiter):
    scores=[(costf(v),v) for v in pop]
    scores.sort()
    ranked=[v for (s,v) in scores]
    
    # Start with the pure winners 从存粹的胜者开始
    pop=ranked[0:topelite]
    
    # Add mutated and bred forms of the winners 添加配对后的胜出者
    while len(pop)<popsize:
      if random.random()<mutprob:

        # Mutation 变异
        c=random.randint(0,topelite)
        pop.append(mutate(ranked[c]))
      else:
      
        # Crossover 交叉
        c1=random.randint(0,topelite)
        c2=random.randint(0,topelite)
        pop.append(crossover(ranked[c1],ranked[c2]))
    
    # Print current best score 最优值
    print (scores[0][0])
    
  return scores[0][1]
