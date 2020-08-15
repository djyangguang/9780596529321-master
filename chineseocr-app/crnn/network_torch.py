#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch ocr model
@author: chineseocr
"""
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from collections import OrderedDict
from torch.autograd import Variable
from crnn.util import resizeNormalize ,strLabelConverter

class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) # 这句话的出现就是为了将前面多维度的tensor展平成一维。
        return output
    

### yg 2020年1月2日13:53:57
"""Pytorch中神经网络模块化接口nn的了解"""
"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
定义自已的网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中，
    不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
    
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
    if,for,print,log等python语法.
    
    注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
    比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：
    
    input_image = torch.FloatTensor(1, 28, 28)
    input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28
    
"""


class CRNN(nn.Module):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        # nn.Conv2d返回的是一个Conv2d class的一个对象，该类中包含forward函数的实现
        # 当调用self.conv1(input)的时候，就会调用该类的forward函数

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False,lstmFlag=True,GPU=False,alphabet=None):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__() #   这个方法中可以封装多个子类，注意，一定继承nn.Module的类，在调用的时候，可以使用下面的方法
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2] # 卷积核大小。
        ps = [1, 1, 1, 1, 1, 1, 0] # 补0 控制zero-padding的数目。
        ss = [1, 1, 1, 1, 1, 1, 1]# 步长 可以设为1个int型数或者一个(int, int)型的tuple。
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag
        self.GPU = GPU
        self.alphabet = alphabet
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            '''layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=0)
        in_channels:个人认为有两种理解
                     1）输入通道数，对于图片层一般为1（灰度）3（RGB）
                     2）定义一种输入规则，要求上一层的输出必须和这个输入一致，也可以理解为并发in_channels个channel在上一层       feature_map(特征映射)上进行卷积运算
        out_channels:

                    1)直观理解是输出层通道数，
                    2)换一种理解是kernels（卷积核）个数，其中，每个卷积核会输出局部特征，比如下图中
                    面部中有头发feature，衣服颜色的feature都是由不同的kernel进行卷积运算得到的。
        '''
            cnn.add_module('conv{0}'.format(i), # https://blog.csdn.net/Haiqiang1995/article/details/90300686
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])) # nn.Conv2d的功能是：对由多个输入平面组成的输入信号进行二维卷积
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut)) #  # 我们知道使用sigmoid会出现梯度消失的情况，在实际训练中，引入了BatchNorm操作，可以将输入值限定在(\gamma ,\beta )之间，
                #如果不进行Batch Norm，如果输入weight差别过大，在两个方向进行梯度下降，会出现梯度下降不平衡，在训练过程中不能稳定的收敛，在实际应用过程中也不能稳定的输出label结果，因此Normalization是很重要的
                #     目前已知的Normalization的方法有4种，对于输入数据为[N,C,(H*W)](N代表tensor数量，C代表通道，H代表高，W代表宽)

                '''其中第一种最为常见
                
                Batch Norm:对每一个批次（N个tensor）的每个通道分别计算均值mean和方差var,如[10,4,9] 最终输出是[0,1,2,3]这样的1*4的tensor
                Layer Norm:对于每一个tensor的所有channels进行均值和方差计算
                Instance Norm:对于每个tensor的每个channels分别计算
                Group Norm：引用了group的概念，比如BGR表示一个组 --不常见'
                '''

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64  下采样 。降维度 在pytorch中使用Pooling操作来实现采样，常见的pool操作包含Max_pool，Avg_pool等

        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))
        else:
            self.linear = nn.Linear(nh*2, nclass)   #  分类器是一个简单的nn.Linear()结构，输入输出都是维度为一的值，x = x.view(x.size(0), -1)  这句话的出现就是为了将前面多维度的tensor展平成一维。
            

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
           # rnn features
           output = self.rnn(conv)
           T, b, h = output.size()
           output = output.view(T, b, -1)
           
        else:
             T, b, h = conv.size()
             t_rec = conv.contiguous().view(T * b, h)
             output = self.linear(t_rec)  # [T * b, nOut]
             output = output.view(T, b, -1)
             
                     # 返回值也是一个Variable对象

        return output
    
    def load_weights(self,path):
        
        trainWeights = torch.load(path,map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.','') # remove `module.`
            modelWeights[name] = v      
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available() and self.GPU:
            self.cuda()
        self.eval()
        
    def predict(self,image):
        image = resizeNormalize(image,32)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available() and self.GPU:
           image   = image.cuda()
        else:
           image   = image.cpu()
            
        image       = image.view(1,1, *image.size()) # 这句话的出现就是为了将前面多维度的tensor展平成一维。
        image       = Variable(image)
        preds       = self(image)
        _, preds    = preds.max(2)
        preds       = preds.transpose(1, 0).contiguous().view(-1)  # 这句话的出现就是为了将前面多维度的tensor展平成一维。
        raw         = strLabelConverter(preds,self.alphabet)  # 标签转换器


        return raw
    
    def predict_job(self,boxes):
        n = len(boxes)
        for i in range(n):
            
            boxes[i]['text'] = self.predict(boxes[i]['img'])
            
        return boxes
     
    def predict_batch(self,boxes,batch_size=1):
        """
        predict on batch
        """

        N = len(boxes)
        res = []
        imgW = 0
        batch = N//batch_size
        if batch*batch_size!=N:
            batch+=1
        for i in range(batch):
            tmpBoxes = boxes[i*batch_size:(i+1)*batch_size]
            imageBatch =[]
            imgW = 0
            for box in tmpBoxes:
                img = box['img']
                image = resizeNormalize(img,32)
                h,w = image.shape[:2]
                imgW = max(imgW,w)
                imageBatch.append(np.array([image]))
                
            imageArray = np.zeros((len(imageBatch),1,32,imgW),dtype=np.float32)
            n = len(imageArray)
            for j in range(n):
                _,h,w = imageBatch[j].shape
                imageArray[j][:,:,:w] = imageBatch[j]
            
            image = torch.from_numpy(imageArray)
            image = Variable(image)
            if torch.cuda.is_available() and self.GPU:
                image   = image.cuda()
            else:
                image   = image.cpu()
                
            preds       = self(image)
            preds       = preds.argmax(2)
            n = preds.shape[1]
            for j in range(n):
                res.append(strLabelConverter(preds[:,j],self.alphabet))

              
        for i in range(N):
            boxes[i]['text'] = res[i]
        return boxes
            
        
            
            
        
            
        
        
        
        