import pandas as pd
import multiprocessing as mp
import time
import math
import json
import numpy as np
import scipy
from scipy.sparse import coo_matrix ,csr_matrix
from tfidfcount import standardlise,splitdataset,read_stopword


def count(nums,dim,csrlist,dotlist,d):
    for i in nums:
        dlist = []
        for j in range(dim):
            dlist.append(csr_matrix.multiply(csrlist[i], csrlist[j]).sum() / (dotlist[i] * dotlist[j]))
        d[i]=dlist

if __name__=='__main__':
    stopwords=read_stopword()
    mat_list = np.load("E:/pythonwork/number.npy")
    csrlist = []
    for i in mat_list:
        coo = coo_matrix(i)  ##稀疏矩阵转化为coo
        csr = coo.tocsr()  # 为了方便运算，转化为csr
        csrlist.append(csr)
    print(len(csrlist))
    documents_st, documents_pri = standardlise(stopwords)
    dim = len(documents_st)
    dotlist = []
    print("===计算模====")
    for i in range(len(documents_st)):
        dotlist.append(scipy.sqrt((csr_matrix.multiply(csrlist[i], csrlist[i])).sum()))
    numset=[x for x in range(len(documents_st))]
    print(len(numset))
    nums=splitdataset(numset,8)
    print("===计算相似度====")
    start = time.clock()
    with mp.Manager() as manager:
        d = manager.dict()
        pool = mp.Pool()
        for num in nums:
            pool.apply_async(count, (num, dim,csrlist,dotlist, d))
        pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
        pool.join()  # 等待进程池中的所有进程执行完毕
        print("Sub-process(es) done.")
        end = time.clock()
        print('Running time: %s Seconds' % (end - start))
        info_json2 = json.dumps(dict(d))
        f = open('E:/pythonwork/similarity.json', 'w')
        f.write(info_json2)
        f.close()
