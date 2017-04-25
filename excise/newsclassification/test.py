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
    if  line :
        tags = jieba.analyse.extract_tags(line,topK=3,withWeight=True)
        l = list(tags)
        for word in l:
            s = word[0].encode('utf-8')
            fre = word[1]
            print(s + str(fre))
        
    
        
        