#!/usr/bin/python
# -*- coding: UTF-8 -*-

# A dictionary of movie critics and their ratings of a small
# set of movies
# -*- coding: UTF-8 -*-
from math import sqrt
prefs={'老杨': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
 'The Night Listener': 3.0},
'老杨01': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 3.5},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0,
 'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'老杨03': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

# 基于物品 的 皮尔逊 距离
# p1 ='老杨'
# p2 = 'Jack Matthews'
def sim_pearson(prefs, p1, p2):
 # Get the list of mutually rated items
 si = {}
 #prefs = {'Snakes on a Plane': 4.5, 'Superman Returns': 4.0, 'You, Me and Dupree': 1.0}
 #item = 'You, Me and Dupree'
 for item in prefs[p1]:
  #prefs[p2] =  {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'You, Me and Dupree': 3.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0}
  if item in prefs[p2]:
   #si[item] ={'Snakes on a Plane': 1, 'Superman Returns': 1, 'You, Me and Dupree': 1}
     si[item] = 1

 # if they are no ratings in common, return 0
 if len(si) == 0: return 0

 # Sum calculations
 n = len(si)

 # Sums of all the preferences
 #  所有偏好和
 '''prefs[p1][it] = 4.5
 # it = 'Snakes on a Plane'
 # si ={'Snakes on a Plane': 1, 'Superman Returns': 1, 'You, Me and Dupree': 1}
 '''
 sum1 = sum([prefs[p1][it] for it in si])
 sum2 = sum([prefs[p2][it] for it in si])
 # Sums of the squares 平方和  xit依次表示si中的一个元素，遍历完所有元素循环结束
 sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
 sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

 # Sum of the products 乘积之和
 pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

 # Calculate r (Pearson score) 皮尔逊 评价值
 num = pSum - (sum1 * sum2 / n)
 den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
 if den == 0: return 0

 r = num / den

 return r


def getRecommendations(prefs, person, similarity=sim_pearson):
 totals = {}
 simSums = {}
 # other =  'Jack Matthews'
 for other in prefs:
  # don't compare me to myself
  if other == person: continue
  sim = similarity(prefs, person, other)

  # ignore scores of zero or lower 去掉 平价<= 0的
  if sim <= 0: continue
  for item in prefs[other]:

   # only score movies I haven't seen yet　只对自己没看过的经行平价
  # item = 'Lady in the Water'
   if item not in prefs[person] or prefs[person][item] == 0:
    # Similarity * Score　相
   # totals ={'Lady in the Water': 0}
    totals.setdefault(item, 0)
    #totals[item] = {'Lady in the Water': 1.9885469410796102}
    totals[item] += prefs[other][item] * sim  # 似度×评价值
    # Sum of similarities
    #simSums =  {'Lady in the Water': 0}
    simSums.setdefault(item, 0)
    #simSums[item] 0.66284898036
    simSums[item] += sim  # 相似度和

 # Create the normalized list 建立一个归一化的列表
 #  simSums ={'Lady in the Water': 2.9598095649952163, 'Just My Luck': 3.190365732076911, 'The Night Listener': 3.853214712436781}
 #  item = 'Lady in the Water'
 # < type
 # (total / simSums[item], item) ='tuple' >: (2.832549918264162, 'Lady in the Water')
 # < type
 # totals.items() = 'list' >: [('Lady in the Water', 8.383808341404684), ('Just My Luck', 8.07475410584156),
 #            ('The Night Listener', 12.89975185847269)]
 rankings = [(total / simSums[item], item) for item, total in totals.items()]

 # Return the sorted list
 # < type
 # 'list' >: [(2.5309807037655645, 'Just My Luck'), (2.832549918264162, 'Lady in the Water'),
 #            (3.3477895267131013, 'The Night Listener')]
 rankings.sort()
 # < type
 # 'list' >: [(3.3477895267131013, 'The Night Listener'), (2.832549918264162, 'Lady in the Water'),
 #            (2.5309807037655645, 'Just My Luck')]
 rankings.reverse()
 return rankings

######利用所有他人平价  值的 加权平均，为某人提供建议
print getRecommendations(prefs,'老杨03')
