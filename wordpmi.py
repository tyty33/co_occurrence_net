#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import os
import re
import numpy as np
import jieba.posseg as psg
import networkx as nx
import pandas as pd
import math
os.chdir("D:/python_ex/word_pmi")


# In[2]:


def get_stop_dict(file):
    content = open(file,encoding="utf-8")
    word_list = []
    for c in content:
        c = re.sub('\n|\r','',c)
        word_list.append(c)
    return word_list


# In[3]:


def get_data(path):
    t = open(path,encoding="utf-8")
    data = t.read()
    t.close()
    return data


# In[4]:


def get_wordlist(text,maxn,synonym_words,stop_words):
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n','nz','vn']#a,形容词，v,形容词
    counts={}
    
    
    text_seg = psg.cut(text)
    for word_flag in text_seg:
        #word = re.sub("[^\u4e00-\u9fa5]","",word_flag.word)
        word = word_flag.word
        if word_flag.flag in flag_list and len(word)>1 and word not in stop_words:
            if word in synonym_origin:
                index = synonym_origin.index(word)
                word = synonym_new[index]
            counts[word]=counts.get(word,0)+1
            
    
    words= sorted(counts.items(),key=lambda x:x[1],reverse=True)
    words= list(dict(words).keys())[0:maxn]
    
    return words


# In[5]:


def get_t_seg(topwords,text,synonym_words,stop_words):
    word_docs = {}
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n','nz','vn']#a,形容词，v,形容词
    
    text_lines_seg =[]
    text_lines = text.split("\n")
    for line in text_lines:
        t_seg = []
        text_seg = psg.cut(line)
        for word_flag in text_seg:
            #word = re.sub("[^\u4e00-\u9fa5]","",word_flag.word)
            word = word_flag.word
            if word_flag.flag in flag_list and len(word)>1 and word not in stop_words:
                if word in synonym_origin:
                    word = synonym_new[synonym_origin.index(word)]
                if word in topwords:
                    t_seg.append(word)
                    
        t_seg=list(set(t_seg))
        for word in t_seg:
            word_docs[word]=word_docs.get(word,0)+1
        text_lines_seg.append(t_seg)
    return text_lines_seg,word_docs


# In[6]:


def get_comatrix(text_lines_seg):
    comatrix = pd.DataFrame(np.zeros([len(topwords),len(topwords)]),columns=topwords,index=topwords)
    for t_seg in text_lines_seg:
        for i in range(len(t_seg)-1):
                for j in range(i+1,len(t_seg)):
                    comatrix.loc[t_seg[i],t_seg[j]]+=1
    for k in range(len(comatrix)):
        comatrix.iloc[k,k]=0
    return comatrix


# In[7]:


def get_pmi(word1,word2,word_docs,co_matrix,n):
    pw1 = word_docs[word1]/n
    pw2 = word_docs[word2]/n
    pw1w2 = (co_matrix.loc[word1][word2]+co_matrix.loc[word2][word1])/n
    if pw1w2/(pw1*pw2)<=0:
        return 0
    else:
        pmi = math.log2(pw1w2/(pw1*pw2))
        return pmi


# In[8]:


def get_net(copmi,topwords):
    g = nx.Graph()
    for i in range(len(topwords)-1):
        word = topwords[i]
        for j in range(i+1,len(topwords)):
            w=0
            word2 = topwords[j]
            w = copmi.loc[word][word2]+copmi.loc[word2][word]
            if w>0:
                g.add_edge(word,word2,weight=w)
    return g


# In[9]:


#文件路径
dic_file = "./stop_dic/dict.txt"
stop_file = "./stop_dic/stopwords.txt"
data_path = "./data/data.txt"
synonym_file = "./stop_dic/synonym_list.xlsx"


# In[10]:


#读取文件
data = get_data(data_path)
stop_words = get_stop_dict(stop_file)
jieba.load_userdict(dic_file)
synonym_words = pd.read_excel(synonym_file)


# In[11]:


#数据处理
n_topwords=200
topwords = get_wordlist(data,n_topwords,synonym_words,stop_words)


# In[12]:


t_segs,word_docs = get_t_seg(topwords,data,synonym_words,stop_words)
n = len(t_segs)
co_matrix = get_comatrix(t_segs)


# In[13]:


copmi = pd.DataFrame(np.zeros([len(topwords),len(topwords)]),columns=topwords,index=topwords)
for i in range(len(topwords)-1):
    word1 = topwords[i]
    for j in range(i+1,len(topwords)):
        word2 = topwords[j]
        copmi[word1][word2] = get_pmi(word1,word2,word_docs,co_matrix,n)


# In[14]:


co_net =get_net(copmi,topwords)


# In[15]:


nx.write_gexf(co_net,"./result/word_pmi.gexf")


# In[ ]:




