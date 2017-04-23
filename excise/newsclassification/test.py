# -*- coding: utf-8 -*
'''
Created on 2017 

@author: xiaoyuan发的是
'''


import jieba
import jieba.analyse




f = open("data/train","r")
while True:
    line = f.readline()
    tags = jieba.analyse.extract_tags(line,topK=3)
    l = list(tags)
    for word in l:
        s = word.encode('utf-8')
        print(s )
    
        
        