#!/usr/bin/python
# -*- coding: UTF-8 -*-
#from recommendations import critics
#print critics['Lisa Rose']['Lady in the Water']
import sys

# sys.path.append("E:\pydd\9780596529321-master\PCI_Code Folder\chapter2")
# reload(chapter2\recommendations)
import recommendations

# 原因是因为在  reload某个模块的时候，需要先import来加载需要的模块，这时候再去reload就不会有问题，具体看下面代码:


reload(recommendations)
#print recommendations.sim_distance(recommendations.critics,'老杨','老杨01')
#print recommendations.topMatches(recommendations.critics,
             #                    '老杨',n=5)


print recommendations.getRecommendations(recommendations.critics,'老杨03')
