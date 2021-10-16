import re
import math
import numpy as np
import json
import pandas as pd
import scipy
from scipy.sparse import coo_matrix, csr_matrix
import sys
import time


def open_stop(filename):  # 打开停用词表
    stopwords = []
    with open(filename) as s:
        while True:
            line = s.readline()
            if line == '':
                break
            line = line.strip()
            stopwords.append(line)
    return stopwords


def open_ori_file(filename, stopwords):  # 打开语料，建立词袋
    wordbags = []
    with open(filename, encoding='gbk') as f:
        file = f.read()
        word_ = re.findall("([^a-zA-Z].*?)/", file)
        for str_ in word_:
            str_ = str_.strip()
            if (str_ not in stopwords) and len(str_)!=19 and str_!='':  # 文本分类的标号为19个字符
                wordbags.append(str_)
    wordbags = list(set(wordbags))
    print("词袋中的数量是：", len(wordbags))
    #print(wordbags)
    return wordbags


def concat_doc(filename, stopwords):  # 把同一篇文章拼接起来
    documents = []
    with open(filename, encoding='gbk') as f:
        file = f.readlines()
        file_ = []
    for doc in file:
        word_ = re.findall("([^a-zA-Z].*?)/", doc)
        if word_:
            file_.append(word_)
    temp = []
    for i in range(len(file_)-1):
        temp.extend(file_[i])
        if file_[i][0][:-4] == file_[i+1][0][:-4]:
            continue
        else:
            documents.append(temp)
            temp = []
    print("拼接后文章数: %d 个 " % (len(documents)))
    documents_st = []
    for doc in documents:
        wordlist = []
        for str_ in doc:
            str_ = str_.strip()
            if (str_ not in stopwords) and len(str_) != 19 and str_ != "":
                wordlist.append(str_)
        documents_st.append(wordlist)
    print("第一篇文章：")
    print(documents_st[0])
    documents_pri = []
    for doc in documents:
        s = ""
        for i in doc:
            if len(i) != 19:
                s = s + i.strip()
        documents_pri.append(s)
    return documents_st


def concat_doc2(filename, stopwords):  # 把同一篇文章拼接起来并且合并成连续的一整篇
    documents = []
    with open(filename, encoding='gbk') as f:
        file = f.readlines()
        file_ = []
    for doc in file:
        word_ = re.findall("([^a-zA-Z].*?)/", doc)
        if word_:
            file_.append(word_)
    temp = []
    for i in range(len(file_)-1):
        temp.extend(file_[i])
        if file_[i][0][:-4] == file_[i+1][0][:-4]:
            continue
        else:
            documents.append(temp)
            temp = []
    documents_pri = []
    for doc in documents:
        s = ""
        for i in doc:
            if len(i) != 19:
                s = s + i.strip()
        documents_pri.append(s)
    return documents_pri


def coutfreq(doc, num=0):  # 计算tf，统计该文档内的词频 num参数可以选模式，默认0为普通词频，1为除以最大词频，2为除以文本长度
    worddict = {}
    for word in doc:
        if word in worddict:
            worddict[word] += 1
        else:
            worddict[word] = 1
    if num == 0:
        return worddict
    elif num == 1:
        maxtf = max(worddict.values())
        for k in worddict.keys():
            worddict[k] = (worddict[k]/maxtf)
        return worddict
    elif num == 2:
        for k in worddict.keys():
            worddict[k] = (worddict[k]/len(worddict))
        return worddict


def countfreq_idf(wordbags, documents):
    dict_idf = {}
    for word in wordbags:
        num = 0
        for doc in documents:
            if word in doc:
                num += 1
        idf = math.log(len(documents)/(num+1))
        dict_idf[word] = idf
    return dict_idf


def countf_tf(word, dict):
    return dict[word]


def countf_idf(word, idict):
    return idict[word]


def count_tfidf(word, doc_index, tflist, dict_idf):
    """
       word 查询词
       doc_index 查询文章在文档集中索引
       tflist_1f tf词频字典列表
       dict_idf idf字典
    """
    dict_tf = tflist[doc_index]
    tf = dict_tf[word]
    idf = dict_idf[word]
    return tf*idf


def string_hash(source):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7
        m = 1000003
        mask = 2 ** 128 - 1
        for c in source:
            x = ((x * m) ^ ord(c))
        x ^= len(source)
        if x == -1:
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]
        print(source, x)
        return str(x)


def hamming_distance(x, y):
    x=int(x,2)
    y=int(y,2)
    return bin(x ^ y).count('1')


def comp_doc(x, y, s_h):
    return hamming_distance(s_h[x], s_h[y])


if __name__ == '__main__':
    file1 = 'E:/pythonwork/停用词.txt'
    file2 = 'E:/pythonwork/199801_clear .txt'
    stopwords = open_stop(file1)
    wordsbags = open_ori_file(file2, stopwords)
    documents_st = concat_doc(file2, stopwords)
    tflist_1 = []  # 标准化tf
    tflist_0 = []  # 词频tf
    tflist_2 = []  # 文章长度修正tf
    for doc in documents_st:
        tflist_1.append(coutfreq(doc, 1))
        # tflist_2.append(coutfreq(doc, 2))
        # tflist_0.append(coutfreq(doc, 0))
    f2 = open('E:/pythonwork/idf_dict.json', 'r')
    dict_idf = json.load(f2)
    f2.close()

    # dict_idf = countfreq_idf(wordsbags, documents_st)  # 计算idf的字典
    # test
    # 第一篇文章中标准化后前五十的关键词
    '''
    d = {}
    for i in documents_st[0]:
        d[i] = count_tfidf(i, 0, tflist_1, dict_idf)
    d_order = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print("第一篇文章中标准化后前五十的关键词：")
    print(d_order[:50])
    '''
    # test end
    s_h = []
    for doc in range(len(documents_st)):
        t = {}
        for i in documents_st[doc]:
            t[i] = count_tfidf(i, doc, tflist_1, dict_idf)
        t_order = sorted(t.items(), key=lambda x: x[1], reverse=True)
        # print(t_order[:20])
        keyList = []
        klen=min(20,len(t_order))
        for m in range(klen):
            feature = string_hash(t_order[m][0])
            weight = t_order[m][1]
            temp = []
            for i in feature:
                if i == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            keyList.append(temp)
        list1 = np.sum(np.array(keyList), axis=0)
        sim_hash = ''
        if not keyList:
            sim_hash = '00'
        for i in list1:
            if i > 0:
                sim_hash = sim_hash + '1'
            else:
                sim_hash = sim_hash + '0'
        print(str(doc)+'篇文章：')
        # print(sim_hash)
        s_h.append(sim_hash)
        np.save("E:/pythonwork/reuslt_hash.npy", s_h)
    # for i in range(len(s_h)-1):
    #     for m in range(len(s_h)-1):
    #         comp_doc(i, m, s_h)
