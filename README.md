# Evaluation-Object-Extracted
**基于深度学习的评价对象提取**

##模型
![](https://github.com/yangzhiye/ImageCache/blob/master/eoe&eosc/eoe.png?raw=true)

##结果    
* eoe_lstm_model accuracy is 0.568681

* eoe_GRU_model accuracy is 0.565934

* eoe_Blstm_model accuracy is 0.527473


# Evaluation-Object-Semantic-Classification
**对提取到的评价对象做情感分析**

##模型1:lstm
![](https://github.com/yangzhiye/ImageCache/blob/master/eoe&eosc/eosc.png?raw=true)
**本模型来源于论文 《Effective LSTMs for Target-Dependent Sentiment Classification》**

##模型2:Bilstm
**在模型1的基础上进行了改进,使用Bilstm替代lstm,return_sequences置true,将所有输出mean之后再concat，关键代码如下:**


![](https://github.com/yangzhiye/ImageCache/blob/master/eoe&eosc/eosc2.png?raw=true)


**from train_model.py**

##结果
* eosc_lstm_model accuracy is 0.596154
  
* eosc_Blstm_model accuracy is 0.604396

##备注
**模型输入数据中的词向量不好，是我为了省内存用train和test集少量数据训练的**

**如果更换成别人用大语料训练的word2vec或者glove，用多层神经网络或者在训练模型的过程中继续训练词向量的话效果会更好**
