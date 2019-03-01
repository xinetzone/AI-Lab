## 数据挖掘

一本很不错的书籍 [Data Mining and Analysis](http://www.dataminingbook.info/pmwiki.php/Main/BookResources), 我已经下载到: [books](books/DataMining-and-Analysis.pdf), 同时也提供了该书的数据集[下载链接](http://www.dataminingbook.info/pmwiki.php/Main/BookPathUploads?action=download&upname=datasets.zip). 数据集的使用方法见: [几个简单数据集预览.ipynb](几个简单数据集预览.ipynb)

### 监督学习

监督学习任务的重点在于：根据已有的经验知识对未知样本的目标/标记进行预测。根据目标预测变量的不同可以将监督学习任务大体分为：分类和回归。监督学习的一般流程：

- [ ] 1. 准备训练数据，可以是文本、图像、音频等；
- [ ] 2. 抽取训练数据的特征，形成特征向量；
- [ ] 3. 将特征向量连同其对应的标记/目标一并送入学习算法中，训练出一个预测模型；
- [ ] 4. 采用同样的特征抽取方法作用于测试数据，得到用于测试的特征向量；
- [ ] 5. 使用预测模型对那些待测试的特征向量进行预测并得到结果。

#### 分类

- [线性分类器](分类/LogisticsRegression.ipynb)
- [SVC](分类/SVC.ipynb)
- [朴素贝叶斯](分类/朴素贝叶斯.ipynb)
- [kNN 分类](分类/kNNC.ipynb)
- [决策树](分类/决策树.ipynb)
- [集成分类](分类/集成分类.ipynb)
- [softmax](分类/softmax.ipynb)