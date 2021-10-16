import re
import multiprocessing as mp
import time
import math
import json
import numpy as np
import scipy
from scipy.sparse import coo_matrix ,csr_matrix
import sys

# 读取停用词
def read_stopword(path="E:/pythonwork/停用词.txt"):
    stopwords = []
    with open(path) as g:
        while True:
            line = g.readline()
            if line == "":
                break
            line = line.strip()
            stopwords.append(line)
    return stopwords

def construt_wordbags(stopwords):
    start = time.clock()
    wordbags = []
    with open("E:/pythonwork/199801_clear .txt") as f:
        file = f.read()
        word_ = re.findall("([^a-zA-Z].*?)/", file)  # 匹配斜线前的词语
        for strss in word_:
            strss = strss.strip()
            if (strss not in stopwords) and len(strss) != 19 and strss != "":
                wordbags.append(strss)
    wordbags = list(set(wordbags))
    print("词袋中的数量是: ", len(wordbags))
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    return wordbags

def standardlise(stopwords):
    with open("E:/pythonwork/199801_clear .txt") as f:
        documents = f.readlines()
        documents_ = []
        for doc in documents:
            doc_ = re.findall("([^a-zA-Z].*?)/", doc)  # 匹配斜线前的词语
            if (doc_):
                documents_.append(doc_)
        temp = []
        documents_2 = []
        for i in range(len(documents_) - 1):
            temp.extend(documents_[i])
            if (documents_[i][0][:-4] == documents_[i + 1][0][:-4]):
                continue
            else:
                documents_2.append(temp)
                temp = []
    print("拼接后文章数: %d 个 " % (len(documents_2)))
    documents_st = []
    for doc in documents_2:
        wordlist = []
        for strss in doc:
            strss = strss.strip()
            if (strss not in stopwords) and len(strss) != 19 and strss != "":
                wordlist.append(strss)
        documents_st.append(wordlist)
        documents_pri = []
    for doc in documents_2:
        s = ""
        for i in doc:
            if (len(i) != 19):
                s = s + i.strip()
        documents_pri.append(s)
        return documents_st,documents_pri

def countfreq_idf(wordbags,documents,dict_idf):
    for word in wordbags:
        num=0
        for doc in documents:
            if word in doc:
                num+=1
        idf = math.log(len(documents)/(num+1))
        dict_idf[word]=idf
    return dict_idf

def countfreq_idf2(wordbags,documents):
    dict_idf={}
    for word in wordbags:
        num=0
        for doc in documents:
            if word in doc:
                num+=1
        idf = math.log(len(documents)/(num+1))
        dict_idf[word]=idf
    return dict_idf

#切分数据集成n份
def splitdataset(listTemp, n):
    resList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        resList[i % n].append(e)
    return resList

#tf-idf算法
##1.统计tf词频
def coutfreq(doc,num=0): #统计该文档内的词频 num参数可以选模式，默认0为普通词频，1为除以最大词频，2为除以文本长度
    worddict={}
    for word in doc:
        if word in worddict:
            worddict[word]+=1
        else:
            worddict[word]=1
    if(num==0):
        return worddict
    elif(num==1):
        maxtf= max(worddict.values())
        for k in worddict.keys():
            worddict[k]=(worddict[k]/maxtf)
        return worddict
    elif(num==2):
        for k in worddict.keys():
            worddict[k]=(worddict[k]/len(worddict))
        return worddict

def countf_tf(word,dict): #根据map查值
    return dict[word]

def countf_idf(word,idict):
    return idict[word]

def count_tfidf(word,doc_index,tflist,dict_idf):
    '''word 查询词
       doc_index 查询文章在文档集中索引
       tflist_1f tf词频字典列表
       dict_idf idf字典
    '''
    dict_tf=tflist[doc_index]
    tf=dict_tf[word]
    idf=dict_idf[word]
    return tf*idf

def build_matrix(num,documents_st,tflist,dict_idf,wordbags):
    matlist=[]
    for index in num:
        m=[count_tfidf(word,index,tflist,dict_idf) if word in documents_st[index] else 0 for word in wordbags]
        sp_mat=np.array(m)
        matlist.append(sp_mat)
    return matlist

if __name__=='__main__':
    stopwords=read_stopword("E:/pythonwork/停用词.txt")
    wordbags=construt_wordbags(stopwords)
    documents_st,documents_pri=standardlise(stopwords)
    # print(documents_st[0])
    wordbag_splitlist=splitdataset(wordbags,10)

    # 读取idf并转化为字典
    f2 = open('E:/pythonwork/idf_dict.json', 'r')
    idf_dict = json.load(f2)
    print(len(idf_dict))
    f2.close()

    # 多进程计算优化计算idf
    start = time.clock()
    print("开始计算idf===============")
    pool = mp.Pool();
    results = []
    for wordset in wordbag_splitlist:
        results.append(pool.apply_async(countfreq_idf2, (wordset,documents_st)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    idf_dict={}
    print(len(results))
    for i in results:
        idf_dict.update(i.get())
    info_json = json.dumps(idf_dict, sort_keys=False, indent=4, separators=(',', ': '))
    f = open('E:/pythonwork/idf_dict.json', 'w')
    f.write(info_json)
    f.close()
    # 显示数据类型
    print(type(info_json))

    tflist_1 = []  ##标准化tf
    tflist_0 = []  ##词频tf
    tflist_2 = []  ##文章长度修正tf
    for doc in documents_st:
        tflist_1.append(coutfreq(doc, 1))
        tflist_2.append(coutfreq(doc, 2))
        tflist_0.append(coutfreq(doc, 0))
    numset=[x for x in range(len(documents_st))]
    nums=splitdataset(numset,8)
    print(nums)

    start = time.clock()
    print("=====开始生成文章向量======")
    pool = mp.Pool();
    results = []
    for num in nums:
        results.append(pool.apply_async(build_matrix, (num,documents_st,tflist_1,idf_dict,wordbags)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    mat_list=[]
    print(len(results))
    for i in results:
        mat_list.extend(i.get())
    np.save("E:/pythonwork/number.npy", mat_list)