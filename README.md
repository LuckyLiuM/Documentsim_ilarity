# Document_similarity
 手动编写TF-IDF、simhash等并基于不同距离度量计算文本相似度



### 设计思路：

​	文本相似度计算方法有 2 个子任务,即文本表示模型和相似度度量方 法，文本表示模型将文本表示为可以计算的数值向量，根据特征构建词向量； 相似度计量方法可以根据词向量计算文本之间的相似度。本文实现了 tfidf 生 成权重向量、simhash 等构建词向量的方法、分别使用余弦相似度、海明距离 进行计算相似度。并利用多进程方法进行优化，最后对各种方法进行了小结。



### 注意问题

详情内容请看 实验报告pdf

数据集和停用词文件在data文件夹内

并行计算的缓存文件请事先跑一遍生成，具体详见代码注释

