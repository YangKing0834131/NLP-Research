Attention

 Github不渲染公式，文中有数学公式，为了正常显示公式，在chrome的扩展程序中，打开chrome网上应用店，然后搜索MathJax Plugin for Github，下载该插件，并且启用，就可以让上述公式正常显示。



## 6 Language models and RNNs

###  语言模型

​     假如我们输入一些文本$x^{(1)}$,...,$x^{(T)}$，那么语言概率模型为：
$$
\begin{aligned} P\left(\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(T)}\right) &=P\left(\boldsymbol{x}^{(1)}\right) \times P\left(\boldsymbol{x}^{(2)} | \boldsymbol{x}^{(1)}\right) \times \cdots \times P\left(\boldsymbol{x}^{(T)} | \boldsymbol{x}^{(T-1)}, \ldots, \boldsymbol{x}^{(1)}\right) \\ &=\prod_{t=1}^{T} P\left(\boldsymbol{x}^{(t)} | \boldsymbol{x}^{(t-1)}, \ldots, \boldsymbol{x}^{(1)}\right) \end{aligned}
$$


### n-gram 语言模型

unigrams: the, students,opened, their·
bigrams: the students, students opened, opened their
trigrams: the students opened, students opened their
4-grams: the students opened their

n-gram 语言模型假设：$x^{(t+1)}$仅依赖于之前n-1个单词，

假设：

​                                  $$P\left(\boldsymbol{x}^{(t+1)} | \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(1)}\right)=P\left(\boldsymbol{x}^{(t+1)} | \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(t-n+2)}\right)$$

条件命题概率：

![1566977991769](./img/1566978018192.png)

统计近似：
$$
\approx \frac{\operatorname{count}\left(\boldsymbol{x}^{(t+1)}, \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(t-n+2)}\right)}{\operatorname{count}\left(\boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(t-n+2)}\right)}
$$

### 神经语言模型

输入：序列化单词$x^{(1)}$,$x^{(2)}$,...,$x^{(t)}$

输出：预测一下个单词概率分布$P\left(\boldsymbol{x}^{(t+1)} | \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(1)}\right)$



#### 固定窗口的神经语言模型

![1566978562494](./img/1566978562494.png)

相比n-gram模型改进

1、没有稀疏化问题

2、不用保存全部的n-gram信息

存在问题

1、$x^{(1)}$与$x^{(2)}$与不同权值矩阵W相乘，输入不对称，不能解决边长输入的问题。

2、扩大固定窗口，扩大权值矩阵W

3、固定窗口大小不够



#### RNN语言模型

![1566979309194](./img/1566979309194.png)

RNN优点

1、可以解决边长输入问题

2、T时刻可以得到来自T-1时刻之前信息

3、输入长短不会影响模型大小

4、在T时刻，权值矩阵值相同

RNN语言模型存在梯度消失问题,变体RNN语言模型

LSTM

GRU

multi-layer

bidirection

##### 训练RNN语言模型

序列单词（平行语料库）$x^{(1)}$,...,$x^{(T)}$

RNN-LM输出分布：$\hat{\boldsymbol{y}}^{(t)}$

交叉熵损失函数：
$$
J^{(t)}(\theta)=C E\left(\boldsymbol{y}^{(t)}, \hat{\boldsymbol{y}}^{(t)}\right)=-\sum_{w \in V} \boldsymbol{y}_{w}^{(t)} \log \hat{\boldsymbol{y}}_{w}^{(t)}=-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}
$$
平均交叉熵损失函数：
$$
J(\theta)=\frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)=\frac{1}{T} \sum_{t=1}^{T}-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}^{(t)}}
$$
![1566979777956](./img/1566979777956.png)

##### 反向传播梯度更新

$$
\frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{\boldsymbol{h}}}=\left.\sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \boldsymbol{W}_{\boldsymbol{h}}}\right|_{(i)}
$$

![1566980071041](./img/1566980071041.png)

### 评估语言模型



语言模型评价指标困惑度perplexity，语言模型越好，下一个词预测概率越大，困惑度越小
$$
\text { perplexity }=\prod_{t=1}^{T}\left(\frac{1}{P_{\mathrm{LM}}\left(\boldsymbol{x}^{(t+1)} | \boldsymbol{x}^{(t)}, \ldots, \boldsymbol{x}^{(1)}\right)}\right)^{1 / T}
$$
等价于交叉熵损失指数函数：
$$
\prod_{t=1}^{T}\left(\frac{1}{\hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}}\right)^{1 / T}=\exp \left(\frac{1}{T} \sum_{t=1}^{T}-\log \hat{\boldsymbol{y}}_{\boldsymbol{x}_{t+1}}^{(t)}\right)=\exp (J(\theta))
$$




### 语言模型任务

预测输入、语音识别、手写识别、拼写/语法修正、身份识别、机器翻译、摘要、对话



语言识别

![1566980814247](./img/1566980814247.png)



句子预测分类：

![1566980857045](./img/1566980857045.png)

问答：

![1566980920460](./img/1566980920460.png)