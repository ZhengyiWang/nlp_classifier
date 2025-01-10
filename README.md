1. emotion_classifier.py使用jieba分词对输入的文本进行分词，删除停用词及标点后，使用TfidfVectorizer函数体积文本的特征向量后，使用分类器对文本进行分类。<br>
2. emotion_predict.py则基于上述训练的结果，用于后续的结果预测、
3. snorkel_emotion.py基于snorkel框架，可以基于规则和投票结果或极大似然估计结果对文本进行分类打标。由于框架属于生成式模型，因此覆盖率和准确率较低。框架主要用于解决样本量过少，无法支持分类器训练的问题。
 <br><br>
参考文献<br>
* https://cloud.tencent.com/developer/article/1876085
*  https://www.cnblogs.com/jimlau/p/13589908.html
