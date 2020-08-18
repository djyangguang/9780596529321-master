#!/usr/bin/python
# -*- coding: UTF-8 -*-
#from recommendations import critics
#print critics['Lisa Rose']['Lady in the Water']
import sys

# sys.path.append("E:\pydd\9780596529321-master\PCI_Code Folder\chapter2")
# reload(chapter2\recommendations)
import recommendations

# 原因是因为在reload某个模块的时候，需要先import来加载需要的模块，这时候再去reload就不会有问题，具体看下面代码:


reload(recommendations)
#print recommendations.sim_distance(recommendations.critics,'老杨','老杨01')
#print recommendations.topMatches(recommendations.critics,
             #                    '老杨',n=5)

#基于用户的推荐
#print recommendations.getRecommendations(recommendations.critics,'老杨03')
#[(3.3477895267131013, 'The Night Listener'), (2.832549918264162, 'Lady in the Water'), (2.5309807037655645, 'Just My Luck')]

#基于物品的推荐
moivies = recommendations.transformPrefs(recommendations.critics)
print recommendations.topMatches(moivies,'Superman Returns')
#[(0.6579516949597695, 'You, Me and Dupree'), (0.4879500364742689, 'Lady in the Water'), (0.11180339887498941, 'Snakes on a Plane'), (-0.1798471947990544, 'The Night Listener'), (-0.42289003161103106, 'Just My Luck')]
