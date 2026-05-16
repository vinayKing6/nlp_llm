---
tags:
  - nlp
  - cs224N
date: 2026-05-16
---
# 阅读材料

[[cs224n-2026-lecture02-wordvecs.pdf]]
[[cs224n_winter2023_lecture1_notes_draft.pdf]]
[[cs224n-2019-notes02-wordvecs2.pdf]]
Suggested Readings:
1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)
3. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)
4. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
5. [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)

Additional Readings:
1. [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
2. [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
3. [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

# 如何表达一个词的含义？

## **Meaning**

- 词语、短语等所代表的含义。 _(The idea that is represented by a word, phrase, etc.)_
    
- 一个人想要通过言语、符号等表达的想法。 _(The idea that a person wants to express by using words, signs, etc.)_
    
- 在文学作品、艺术作品等中所表达的主旨/意蕴。 _(The idea that is expressed in a work of writing, art, etc.)_
    
语言学中理解意义最常见的方式：

![[Pasted image 20260514143705.png|500]]

**核心公式：能指与所指**

图片上方给出了一个等式：

signifier (symbol) $\Leftrightarrow$ signified (idea or thing)

- 能指 (Signifier / Symbol)： 指的是语言的形式，比如单词的拼写（t-r-e-e）或发声。
    
- 所指 (Signified / Idea or thing)： 指的是这个符号所代表的概念或具体的实物。
    
- 指称语义学 (Denotational Semantics)： 这种观点认为，一个词的“意义”就在于它所**指代**（Denote）的那个对象。
    
**具体的例子**

图片下方用“树”形象地展示了这一过程：

tree $\Leftrightarrow$ {🌳, 🌲, 🌴, ...}

- 左侧的 "tree"： 是一个抽象的文本符号。
    
- 右侧的集合：代表现实世界中所有属于“树”范畴的实体。

再来看一个句子：“Zuko makes the tea for his uncle”

在这里，“zuko”这个词是一个符号，一个代表了某个（现实或虚构）世界中名为“zuko”的实体的象征。而“tea”这个词同样也是一个符号，它指向一个所指（signified）的对象——也许是某次具体的沏茶实例。

如果换一种说法：“Zuko likes to make tea for his uncle" 请注意，此时“zuko”依然指代zuko本人，但“tea”现在指代的是一个更广泛的类别——泛指所有的茶，而非某次具体的、那壶热腾腾的美味之水。

现在考虑以下两句话：

Zuko makes the coffee for his uncle. 
Zuko makes the drink for his uncle.
    
哪一句与关于“tea”的那句话“更像”呢？“drink”可能指茶（也可能完全不同！），而“coffe”肯定不是茶，但它与茶又确实很相似，不是吗？

此外，“zuko”与“uncle”是否相似，是因为它们都描述“人”？“这（the）”与“他的（his）”是否相似，是因为它们都从一个类别中挑选出了具体的实例？
# WordNet

此前最常用的 NLP（自然语言处理）解决方案：使用如 WordNet 之类的工具，这是一种包含同义词集（synonym sets）和上位词（hypernyms，即‘is a’关系）列表的词库。

![[Pasted image 20260514144616.png]]

## 像WordNet这样的资源存在的问题

- 资源虽有用但缺乏细微差别：
    例如，“proficient”（精通）被列为“good”（好）的同义词。这仅在某些语境下是正确的。
- 此外，WordNet 在某些同义词集中列出了冒犯性的词汇，却完全没有涵盖这些词的内涵或适用性。
    
- 缺失词汇的新含义：
    例如：wicked, badass, nifty, wizard, genius, ninja, bombest（注：这些词在俚语中常有“酷”、“厉害”等新义）。
        
- 无法做到实时更新！
    
- 具有主观性。
    
- 需要人工劳动来创建和调整。
    
- 无法用于准确计算词语相似度。

# one-hot编码

在传统 NLP 中，我们将词视为离散符号：例如 hotel（酒店）、conference（会议）、motel（汽车旅馆）——这被称为**局部表示**（localist representation）。

此类词汇符号可以用 **one-hot 向量**来表示：

`motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]`

`hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]`

- **向量维度** = 词汇表中的单词总数（例如：500,000+）
    
- **含义**：向量中只有一个 1，其余全为 0。

## one-hot 编码的问题

示例：在网页搜索中，如果用户搜索“Seattle motel”（西雅图汽车旅馆），我们希望能够匹配包含“Seattle hotel”（西雅图酒店）的文档。

- **`motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]`**
    
- **`hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]`**
    

但这两个向量是正交的（orthogonal）。

对于 One-hot 向量而言，不存在天然的相似度概念！

解决方案：

- 可以尝试依赖 WordNet 的同义词列表来获取相似度？
    
- 但众所周知，这种方法效果很差：存在不完整性等问题。
    
- 取而代之的是：学习将相似度直接编码进向量本身。

我们是否应该不把词义表示为“独热向量（one-hot）”，而是表示为一系列特征，以及它与语言学类别和其他单词之间的关系集合？

对于任何一个词，例如 `runners`（奔跑者），我们可以标注极其丰富的信息。有**语法信息**（如：复数名词）；有**派生信息**（如：`runners` 是由动词 `run` 加上一个“执行者”或“施事者”的概念构成的，即“奔跑的人”）；还有**语义信息**（如：`runners` 是“人类”、“动物”或“实体”的**下位词**。所谓下位词是指一种“包含于”的关系，例如：跑步者是一种人）。

目前，英语和其他少数语言中存在大量关于单词标注信息的资源。例如 **WordNet** [Miller, 1995] 标注了同义词、下位词和其他语义关系；**UniMorph** [Batsuren et al., 2022] 则标注了多种语言的形态学（子词结构）信息。利用这些资源，我们可以构建出类似下面这样的词向量：

$$v_{\text{tea}} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \begin{matrix} \text{(复数名词)} \\ \text{(第三人称单数动词)} \\ \text{(饮品的下位词)} \\ \vdots \\ \text{(chai 的同义词)} \end{matrix}$$

但在 2026 年，通过这些方法产生的词向量并非主流，也不会是本课程的重点。其**主要失败原因**在于：

1. **词汇量匮乏**：与能够直接从自然文本源中提取词汇的方法相比，人工标注的资源总是跟不上——更新这些资源成本高昂，且永远不完整。
    
2. **维度与效能的权衡**：要表示所有这些类别，需要极高维度的向量（甚至远超词表大小），而倾向于处理稠密向量的现代神经方法在处理此类向量时表现不佳。
    
3. **数据驱动优于人工定义**：本课程将贯穿一个持续的主题：人类对于“文本应如何表示”的设想，往往不如那些让数据来决定更多细节的方法表现得好——至少在拥有海量学习数据的情况下是如此。

# 通过上下文表示词语

在语言学中，这一想法早在多年前就由 Firth [Firth, 1957] 完美地捕捉到了，他有一句名言：**“观其伴而知其义” (You shall know a word by the company it keeps)**。从高层逻辑来看，你可以将出现在“茶（tea）”这个词周围的单词分布，视为定义该词含义的一种方式。因此，“茶”会出现在 _drank（喝）、the（这）、pot（壶）、kettle（水壶）、bag（包/袋）、delicious（美味的）、oolong（乌龙）、hot（热的）、steam（蒸汽）_ 等词的周围。

显而易见，与“茶”相似的单词（如“咖啡”）也会拥有相似的周围词分布。尽管这一想法看似简单，但它是整个现代 NLP 中最具影响力和最成功的思想之一，其类似物已在无数学习相关领域扎根。

- 分布语义学（Distributional semantics）：一个词的含义是由经常出现在其附近的词决定的。
    
- “道其友，知其人”（语出 J. R. Firth，1957: 11；原意为：通过一个词周围的词来了解这个词）。
    
- 这是现代统计自然语言处理（NLP）中最成功的思想之一！
    
- 当一个词 $w$ 出现在文本中时，它的“**上下文”**（context）是指出现在其附近（在固定大小的窗口范围内）的一组词。
    
- 我们利用 $w$ 出现的多种上下文环境，来构建 $w$ 的表示形式。

![[Pasted image 20260514154643.png]]
# Word vectors

我们将为每个单词构建一个**稠密向量**（dense vector），通过精心选择，使该向量与出现在相似上下文中的单词向量具有相似性；我们使用**向量点积**（标量积）来衡量这种相似度。

**注：** 词向量也被称为（词）嵌入（embeddings）或（神经）词表示。它们属于一种**分布式表示**（distributed representation）。

例如：(随机初始化)
- `banking = [0.286, 0.792, −0.177, −0.107, 0.109, −0.542, 0.349, 0.271]`
    
- `monetary = [0.413, 0.582, −0.007, 0.247, 0.216, −0.718, 0.147, 0.051]`

## Skip-gram Word2vec

Word2vec 是一个用于学习词向量的框架（Mikolov 等人，2013）。

原论文： [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)

**核心思想：**

- 我们拥有一个海量的文本语料库（Corpus）：即一个很长的单词列表。
    
- 固定词汇表中的每个单词都由一个向量表示。
    
- 遍历文本中的每个位置 $t$，该位置包含一个中心词 $c$ 和一些上下文（“外部”）词 $o$。
    
- 利用 $c$ 和 $o$ 的词向量相似度，来计算在给定 $c$ 的情况下出现 $o$ 的概率（反之亦然）。
    
- 不断调整词向量，以最大化这一概率。

![[Pasted image 20260515130756.png|400]]
### Skip-gram 模型 

**核心术语说明**

- **Windows (窗口)：** 指在中心词周围选取的上下文范围（例如左右各取 2 个词）。
    
- **Process (过程)：** 指如何通过模型计算出给定中心词时，出现特定上下文词的概率。
    
- **$P(w_{t+j} | w_t)$：** 在给定中心词 $w_t$ 的条件下，观察到位置为 $t+j$ 的上下文词 $w_{t+j}$ 的条件概率。

![[Pasted image 20260515131201.png]]
![[Pasted image 20260515131212.png]]
### 目标函数 (也叫损失函数)

对于每一个位置 $t = 1, \dots, T$，在给定中心词 $w_t$ 的情况下，预测固定窗口大小 $m$ 内的上下文词。

数据似然概率（Data likelihood）为：

1. 似然概率 (Likelihood)

$$L(\theta) = \prod_{t=1}^{T} \prod_{-m \le j \le m, j \ne 0} P(w_{t+j} | w_t; \theta)$$

- **含义：** 这是衡量模型对整个文本库（长度为 $T$）预测“契合度”的指标。
    
- **数学操作：** 两个 $\prod$ 符号代表**连乘**。它计算的是在所有位置 $t$ 上，给定中心词时预测出其实际上下文词的联合概率。
    
- **$\theta$：** 指所有待优化的变量（在 Word2vec 中就是那些随机初始化的词向量）。
    
2. 目标函数 (Objective Function)

$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} | w_t; \theta)$$

$J(\theta)$ 有时也被称为 **代价函数 (Cost Function)** 或 **损失函数 (Loss Function)**。这里完成了两个关键转换：

- **取对数 ($\log$)：** 将连乘变为**累加** ($\sum$)。这不仅简化了求导运算，还避免了由于概率连乘导致的数值下溢（数值变得极小）。
    
- **取负号 (Negative)：** 因为我们想要最大化似然概率（预测越准越好），而在优化算法中，习惯上是去**最小化**一个函数。通过加负号，我们将“最大化概率”转换成了“最小化损失”。
    
- **除以 $T$：** 取平均值，使得损失函数的大小不随文本总长度的变化而产生剧烈波动。
    
3. 核心结论

	**最小化目标函数 $\Leftrightarrow$ 最大化预测准确率**

	_(Minimizing objective function $\Leftrightarrow$ Maximizing predictive accuracy)_

### 预测函数 (prediction function) ?

- 对于每个词 $w$，我们将使用两个向量：
    
    - 当 $w$ 是**中心词**时，使用向量 $v_w$
        
    - 当 $w$ 是**上下文词**时，使用向量 $u_w$
        
- 那么，对于中心词 $c$ 和上下文词 $o$：
    

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

-  ① 点积用于比较 $o$ 和 $c$ 的相似度：
    
    $u^T v = u \cdot v = \sum_{i=1}^{n} u_i v_i$
    
    点积越大 = 概率越大。
    
-  ② 指数化（Exponentiation）使任何数值变为正数。
    
-  ③ 对整个词汇表进行归一化（Normalize），以生成概率分布。

- 这是 **Softmax** 函数的一个实例 $\mathbb{R}^n \to (0,1)^n$：
    
    $$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)} = p_i$$
    
- Softmax 函数将任意值 $x_i$ 映射到概率分布 $p_i$：
    
    - **“max”：** 因为它会放大最大 $x_i$ 的概率。
        
    - **“soft”：** 因为它仍然会给较小的 $x_i$ 分配一定的概率。
        
    - **在深度学习中被频繁使用。**

### 训练模型：优化参数值以最小化损失

为了训练模型，我们逐渐调整参数以最小化损失（loss）。

- 回想一下：$\theta$ 代表所有的模型参数，整合在一个长向量中。
    
- 在我们的案例中，如果有 $d$ 维向量和 $V$ 个单词，我们得到 $\rightarrow$
    
    $$\theta = \begin{bmatrix} v_{aardvark} \\ v_a \\ \vdots \\ v_{zebra} \\ u_{aardvark} \\ u_a \\ \vdots \\ u_{zebra} \end{bmatrix} \in \mathbb{R}^{2dV}$$
    
- 记住：每个单词都有两个向量。这里的参数规模是 $2 \times d \times V$。因为每个词作为“中心词”时有一个向量 ($v$)，作为“上下文词”时也有一个向量 ($u$)。
    
- 我们通过沿着梯度下降来优化这些参数。
    ![[Pasted image 20260515140510.png|500]]

### 优化算法：梯度下降 (Gradient Descent)

- 我们有一个想要最小化的**代价函数 $J(\theta)$**。
    
- **梯度下降**是一种用于最小化 $J(\theta)$ 的算法。
    
- **核心思想：** 对于当前的 $\theta$ 值，计算 $J(\theta)$ 的梯度，然后朝着**负梯度**的方向迈出一小步。重复此过程。

![[Pasted image 20260515141642.png]]

#### 梯度下降的数学表达

- **更新方程（矩阵形式）：**
    
    $$\theta^{new} = \theta^{old} - \alpha \nabla_{\theta} J(\theta)$$
    
    其中，$\alpha$ = **步长（step size）**或**学习率（learning rate）**。
    
- **更新方程（单个参数形式）：**
    
    $$\theta_j^{new} = \theta_j^{old} - \alpha \frac{\partial}{\partial \theta_j^{old}} J(\theta)$$
    
- **算法逻辑：**
    ```python
    while True:
        theta_grad = evaluate_gradient(J, corpus, theta) # 在整个语料库上计算梯度
        theta = theta - alpha * theta_grad              # 更新参数
    ```

**挑战：自己动手算一遍**
> 3.4 Working through a gradient

[[cs224n_winter2023_lecture1_notes_draft.pdf#page=10&selection=99,0,103,26|cs224n_winter2023_lecture1_notes_draft, 页面 10]]
####  随机梯度下降 (Stochastic Gradient Descent, SGD)

- **问题：** $J(\theta)$ 是语料库中**所有窗口**的函数（可能涉及数十亿个！）。
    
    - 因此，计算 $\nabla_{\theta} J(\theta)$ 的**代价极其昂贵**。
        
    - 在进行单次参数更新之前，你需要等待非常长的时间！
        
- 对于几乎所有的神经网络来说，这都是一个非常糟糕的主意。
    
- **解决方案：随机梯度下降 (SGD)**
    
    - 反复随机抽取窗口，并在每处理一个窗口后就进行更新。
        
- **算法逻辑：**
    ```python
    while True:
        window = sample_window(corpus)                # 随机采样一个窗口
        theta_grad = evaluate_gradient(J, window, theta) # 仅计算该窗口的梯度
        theta = theta - alpha * theta_grad            # 立即更新参数
    ```

**小批量梯度下降 (Mini-batch Gradient Descent)**：介于两者之间，每次抽取一小批数据（如 32 或 64 个窗口）进行更新，这是目前深度学习中最常用的做法。
### Word2vec 参数与计算 

**参数矩阵 (Parameters)**

- **$U$ (outside)：** 上下文词向量矩阵。每一行代表词汇表中一个单词作为“上下文词”时的向量表示。
    
- **$V$ (center)：** 中心词向量矩阵。每一行代表一个单词作为“中心词”时的向量表示。
    
- _注：这就是为什么之前提到每个词有两个向量。_
    ![[Pasted image 20260515142646.png|300]]

**计算过程 (Computations)**

- **$U \cdot v_4^T$ (dot product)：** **点积计算**。这里假设第 4 个单词是当前的中心词（$v_4$）。将上下文矩阵 $U$ 与中心词向量 $v_4$ 相乘，得到一个得分向量。这个向量中的每个值代表了词汇表中对应的词作为该中心词上下文的“相关度”。
    
- **$\text{softmax}(U \cdot v_4^T)$ (probabilities)：** **概率分布**。通过 Softmax 函数，将上述点积得分转换为 (0, 1) 之间的概率值，所有概率之和为 1。
    ![[Pasted image 20260515143242.png|300]]

**词袋模型 (Bag of words model!)**

- **该模型在每个位置做出相同的预测 (The model makes the same predictions at each position)**：
	这意味着在 Skip-gram 模型中，给定一个中心词，它预测距离为 1、2 或 -1、-2 的上下文词时，使用的是**同一套**参数和概率分布。它不关心上下文词的具体位置（是在左边还是右边，是近还是远），只要在窗口内，处理方式都是一样的。
    
- **目标：** 我们希望模型能够对上下文中经常出现的所有单词，都给出合理且较高的概率估计。

![[Pasted image 20260515143534.png]]

这张图片展示了 **Word2vec 训练后的最终成果：高维词向量的空间可视化**。

它直观地证明了 Word2vec 的核心目标：**通过最大化目标函数，将语义相似的词在向量空间中“拉”到一起。**

以下是详细解释：

**核心现象：语义聚类 (Semantic Clustering)**

虽然词向量通常有几百个维度，但通过降维技术（如 t-SNE）将其投影到二维平面后，你会发现：

- **数字类：** 左上角的 _seven, eight, nine, zero_ 紧紧堆在一起。
    
- **日期类：** 左侧中间的 _tuesday, wednesday, thursday_ 形成了一个小簇。
    
- **科学/学术类：** 右下角的 _biology, physics, mathematics, science_ 聚集在一起。
    
- **人名/历史类：** 右侧顶部的 _richard, james, george, paul_ 靠得很近。
    

**为什么会这样？**

这回到了 **分布语义学**（Distributional Semantics）：

- 在语料库中，_tuesday_ 和 _monday_ 经常出现在类似的句子结构中（例如 "I have a meeting on ____"）。
    
- 为了预测概率最大化，模型必须赋予它们相似的向量值，这样它们在空间中的坐标就会非常接近。
    

**空间中的“意义”**

在 Word2vec 空间里，**距离代表相似性**。

- 如果两个词的向量点积（Dot Product）很大，它们在图中的距离就短。
    
- 这意味着模型不仅学会了单词的拼写，还“理解”了它们的类别和属性。
    
以下是该段内容的中文翻译：

# Word2vec 算法族 (Mikolov et al. 2013)：更多细节

- **为什么使用两个向量？** $\rightarrow$ 为了**简化优化过程**。在训练结束时，通常会对这两个向量取平均值。
    
    - _注：也可以实现每词仅用一个向量的算法……这会有一定帮助。_
        
- **两种模型变体：**
    
    1. **Skip-grams (SG)：** 给定中心词，预测上下文（“外部”）词（与位置无关）。
        
    2. **连续词袋模型 (CBOW)：** 根据（词袋形式的）上下文词来预测中心词。
        
- **我们目前介绍了：** Skip-gram 模型。
    
- **训练所用的损失函数：**
    
    1. **原始 Softmax (Naïve softmax)：** 简单但计算代价昂贵（尤其当输出类别极多时）。
        
    2. **更优化的变体：** 例如**层级 Softmax** (Hierarchical Softmax)。
        
    3. **负采样 (Negative Sampling)。**
        
- **到目前为止，我们已经解释了原始 Softmax。**
# Skip-gram 负采样模型

原论文：
[Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)

- **归一化项的计算代价极高**（当输出类别很多时）：
    
    在原始 Softmax 公式中，$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$，分母部分需要对词汇表 $V$ 中的**每一个词**进行求和。如果词汇表有 50 万个词，每一步更新都要计算 50 万次幂运算，速度慢到无法接受。
    
- **解决方案：** 标准的 Word2vec 实现了**带有负采样的 Skip-gram 模型**。
    
- **核心思想：** 训练**二元逻辑回归（Binary Logistic Regression）**，将“真实对”（中心词及其上下文窗口内的词）与若干“噪声对”（中心词与随机抽取的单词）区分开来。
    
## 负采样的数学实现

- **操作过程：**
    
    - 抽取 $k$ 个**负样本**（使用词频概率抽取）。
        
    - **最大化**真实上下文词出现的概率。
        
    - **最小化**随机单词出现在中心词周围的概率。
        
- **目标函数（损失函数）：**
    
    $$J_{neg-sample}(u_o, v_c, U) = -\log \sigma(u_o^T v_c) - \sum_{k \in \{K \text{ sampled indices}\}} \log \sigma(-u_k^T v_c)$$
    
    - **$\sigma$ (Sigmoid 函数)：** $\sigma(x) = \frac{1}{1 + e^{-x}}$。这里使用的是 Sigmoid 而不是 Softmax，将多分类问题简化为了二分类（是上下文 vs 不是上下文）。

	- **正样本项** $−\log⁡\sigma(u_o^Tv_c)$ : 要使 $u_o^Tv_c$​ 尽可能大（点积大 → sigmoid 接近 1 → 损失小）
        
	- **负样本项** $−\log⁡\sigma(−u_k^Tv_c)$ : 因为$\sigma(−x)=1−\sigma(x)$，实际等同于 $−\log⁡(1−\sigma(u_k^Tv_c))$，要使 $u_k^Tv_c$ 尽可能小（负样本与中心词不相似）
        
## 负样本如何抽取

- 使用公式 $P(w) = U(w)^{3/4} / Z$ 进行采样。
    
- **为什么用 3/4 次方？**
    
    这是一个经验公式。它会略微提升**低频词**被抽样到的概率，降低高频词的影响，使得模型对生僻词的捕捉能力更强，训练更均衡。

###  $U(w)$ 是什么？

- **含义：** 它代表 **Unigram 分布**（Unigram Distribution）。
    
- **直观理解：** 在统计学中，这其实就是单词 $w$ 在整个语料库中出现的**原始频率**。
    
- **计算方式：** $U(w) = \frac{\text{单词 } w \text{ 出现的次数}}{\text{语料库中所有单词的总次数}}$。
    
- 如果一个词（比如 "the"）出现得非常频繁，它的 $U(w)$ 就很高。
    
### $P(w)$ 是什么？

- **含义：** 它是**实际进行负采样时使用的概率分布**。
    
- **公式：** $P(w) = \frac{U(w)^{3/4}}{Z}$
    
- **为什么要加 $3/4$ 次方？** 这是一个经验性的“平滑”操作：
    
    - 它会**压缩高频词**的概率（让 "the" 这种词不至于被抽到太多次）。
        
    - 它会**提升低频词**的概率（让生僻词也有机会被模型“看到”）。
        
    - _举例：_ 假设 $U(\text{"rare word"}) = 0.000001$，取 $3/4$ 次方后，它的概率会显著提升，从而增加它作为负样本参与训练的机会。
        
###  $Z$ 是什么？

- **含义：** 它是一个**归一化常数**（Normalization Constant）。
    
- **作用：** 在概率论中，所有可能情况的概率之和必须等于 **1**。
    
- **计算方式：** $Z = \sum_{w \in V} U(w)^{3/4}$。
    
- 因为我们对每个词的原始频率都取了 $3/4$ 次方，处理后的数值加起来就不再是 1 了。所以我们需要把所有处理后的值加在一起得到 $Z$，然后让每个词的分布都除以 $Z$，确保 $P(w)$ 重新成为一个合法的概率分布。

## 带有负采样的随机梯度 

- 在 SGD 中，我们会对每个窗口迭代地提取梯度。
    
- 在每个窗口中，我们最多只有 $2m + 1$ 个中心词/上下文词，加上 $2km$ 个负采样词。因此，梯度 $\nabla_{\theta} J_t(\theta)$ 是**非常稀疏的（very sparse）**！
    
$$\nabla_{\theta} J_t(\theta) = \begin{bmatrix} 0 \\ \vdots \\ \nabla_{v_{like}} \\ \vdots \\ 0 \\ \nabla_{u_I} \\ \vdots \\ \nabla_{u_{learning}} \\ \vdots \end{bmatrix} \in \mathbb{R}^{2dV}$$
该向量绝大部分位置都是 **0**，只有当前窗口涉及到的特定单词（如 $v_{like}, u_I$）才有非零梯度。

- 我们可能**只需要更新实际出现的那些词向量**！
    
- **解决方案：** 你要么需要**稀疏矩阵更新操作**，以仅更新完整嵌入矩阵 $U$ 和 $V$ 中的特定**行（rows）**

- 要么你需要维护一个哈希表（hash）来存储词向量。

- 如果你拥有数百万个词向量并进行分布式计算，**避免传输巨大的更新数据**是非常重要的！

# 共现矩阵（Co-occurrence Matrix)

遍历整个语料库多次（如 Word2vec）似乎有点奇怪；为什么我们不直接统计所有单词附近的数据呢？

- **构建共现矩阵 $X$ 的两种选择：**
    
    - **窗口 (Window)：** 类似于 Word2vec，在每个词周围使用窗口 $\rightarrow$ 捕捉一些语法和语义信息（即“词空间”）。
        
    - **全文档 (Full document)：** 词-文档共现矩阵将给出一般性主题（例如，所有体育术语都会有相似的条目），从而导致**潜在语义分析（LSA）**（即“文档空间”）。
        
我们将这个矩阵称为 $X$。此时，**词嵌入向量** $X_{tea} \in \mathbb{R}^{|V|}$（即矩阵 $X$ 中的一行）比我们之前拥有的旧“独热向量（1-hot）” $e_{tea}$ 要实用得多。

其实我们也可以规定：只有当单词 $w'$ 出现得更近（比如在几个词之内）时，才算作与 $w$ 共现。以下是一个例子，标注了几个相关的共现窗口：

**共现范围越大（例如大窗口或整个文档），产生的表示就越偏向语义甚至主题编码；窗口越短，则越偏向语法编码。**

`“It’s hot and delicious. [I poured [the tea]窗口1 for]窗口3 my uncle.]文档”`

简而言之：

- **短窗口**（如上图中标记为 1 的单词窗口）似乎编码了**语法属性**。例如，名词倾向于紧挨着 `the` 或 `is`；复数名词则不会紧挨着 `a`。
    
- **大窗口**倾向于编码更多的**语义属性**（极端情况下是主题属性）。注意 `poured`（倒）或 `delicious`（美味的）可能离 `tea` 较远，但依然相关。
    
- **文档级窗口**（数千词的长文）直观上通过单词出现在哪类文档（体育、法律、医学等）中来表示词义。
    

我们做的另一个设计决定是：在 $|V|$ 维向量中表示单词的**显式计数**。**这最终被证明是一个巨大的错误。** 高维向量在今天的神经系统中往往难以处理。但另一个问题是：**原始单词计数会过度强调像 `the` 这样极其常见的词的重要性。** 相比之下，取词元频率的对数（log token frequency）要有用得多。

一篇极具影响力的论文通过引入 **GloVe**（Pennington 等人，2014）教会了我们更多关于原始共现方法的弊端。GloVe 是一种基于共现的词表示算法，其效果与 **Word2Vec** 一样出色。然而，由于 Word2Vec 的许多细节在课程后续方法中依然通用，我们将把时间重点放在 Word2Vec 上。

**示例：基于窗口的共现矩阵**
![[Pasted image 20260515154525.png]]

- **窗口长度：** 1（相当于两个单词一起出现，实际常用 5–10）。
    
- **对称性：** 对称矩阵。
    
**示例语料库：**

1. _I like deep learning_
    
2. _I like NLP_
    
3. _I enjoy flying_
    

**表格解析：**

- **"I" 和 "like"** 共现了 2 次（句子 1 和 2）。
    
- **"like" 和 "deep"** 共现了 1 次（句子 1）。
    
- 这种矩阵记录了词汇一起出现的频率。

设 $X$ 表示**词-词共现矩阵（word-word co-occurrence matrix）**，其中 $X_{ij}$ 表示单词 $j$ 出现在单词 $i$ 的上下文中的次数。设 $X_i = \sum_k X_{ik}$ 为任意单词 $k$ 出现在单词 $i$ 上下文中的总次数。最后，设 $P_{ij} = P(w_j|w_i) = \frac{X_{ij}}{X_i}$ 为单词 $j$ 出现在单词 $i$ 上下文中的概率。填充这个矩阵需要对整个语料库进行一次完整的扫描以收集统计数据。对于大规模语料库，这一步的计算开销可能非常大，但这是一次性的前期成本（up-front cost）。

**共现向量的特性与改进**

- **简单的计数共现向量：**
    
    - **规模随词汇量增加：** 词汇量越大，矩阵越大。
        
    - **维度极高：** 需要大量存储空间（尽管它是稀疏的）。
        
    - **鲁棒性差：** 后续分类模型会遇到稀疏性问题，导致模型不够稳健。
        
- **低维向量 (Low-dimensional vectors)：**
    
    - **思路：** 将“大部分”重要信息存储在一个固定的、较小维度的**稠密向量**中。
        
    - **维度：** 通常为 25–1000 维，类似于 Word2vec。
        
    - **关键问题：** 如何降低维度？（通常使用奇异值分解 **SVD** 等技术）。
        
## 经典方法：对矩阵 X 进行降维 (HW1)

对共现矩阵 $X$ 进行**奇异值分解（SVD）**，该方法将矩阵 $X$ 分解为 $U \Sigma V^T$，其中 $U$ 和 $V$ 是单位正交矩阵（即列向量两两正交且模长为 1）。

$$X = U \Sigma V^T$$
其中各部分的含义如下：

- **$U$ ($m \times m$)：** **左奇异向量矩阵**。它的列向量是相互正交的单位向量（标准正交基）。在几何上，它代表了对原始空间的一种**旋转**操作。
    
- **$\Sigma$ ($m \times n$)：** **奇异值矩阵**。这是一个对角矩阵（除了对角线外全是 0），对角线上的元素 $\sigma_1, \sigma_2, \dots$ 被称为**奇异值**。它们按降序排列，代表了矩阵在对应维度上的“能量”或“重要程度”。
    
- **$V^T$ ($n \times n$)：** **右奇异向量矩阵**（$V$ 的转置）。它的行向量同样是标准正交基。在几何上，它代表了另一种**旋转**。

- **操作：** 仅保留前 $k$ 个最大的奇异值（奇异值按大小排列在 $\Sigma$ 矩阵的对角线上），以实现泛化。(具体怎么做的问gpt吧，还挺好算的)
    
- **数学意义：** $\hat{X}$ 是在最小二乘法意义下对 $X$ 的**最佳秩-$k$ 近似**。
    
- **特点：** 这是经典的线性代数结果。但对于大规模矩阵，计算代价非常昂贵。

##  对 X 的工程处理技巧 (Hacks to X)

直接对原始计数运行 SVD 效果并不理想！！！

- **对单元格中的计数进行缩放（Scaling）会有很大帮助：**
    
    - **问题：** 虚词（如 _the, he, has_）出现频率太高 $\rightarrow$ 语法结构对语义的影响过大。
        
    - **解决办法：**
        
        - 对频率取**对数 (log)**。
            
        - 设置上限 $\min(X, t)$，例如 $t \approx 100$。
            
        - 直接忽略这些虚词。
            
- **渐进式窗口（Ramped windows）：** 离中心词越近的词权重越高，越远的词权重越低。
    
- **使用相关性（Correlations）代替原始计数**，然后将负值设为 0。
    
**缩放后的向量中出现了有趣的语义模式**

![[Pasted image 20260515162240.png]]

这张图展示了经过缩放和降维后的向量空间。你会发现向量空间不仅仅是点的堆叠，还捕捉到了**线性关系（Linear Relationships）**：

- **动词与职业的映射：**
    
    - _DRIVE_（驾驶） $\rightarrow$ _DRIVER_（司机）
        
    - _SWIM_（游泳） $\rightarrow$ _SWIMMER_（游泳者）
        
    - _TEACH_（教学） $\rightarrow$ _TEACHER_（教师）
        
- **发现：** 连接动作与执行者的向量在空间中几乎是**平行且长度相等**的。这意味着模型自动学习到了某种语义规律（如“执行者”这一概念）。

# GloVe (Global Vectors for Word Representation)

原论文：
GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)

GloVe 模型由一个加权最小二乘模型（Weighted Least Squares Model）组成，它通过对全局词-词共现计数进行训练，从而有效地利用了统计信息。该模型生成了一个具有显著子结构（meaningful sub-structure）的词向量空间。它在词类比任务上表现出了顶尖水平（state-of-the-art），并在多个单词相似度任务中优于其他方法。

![[Pasted image 20260516151919.png]]
这份幻灯片展示的是 **GloVe (Global Vectors)** 词嵌入模型的核心数学推导。它的核心目标是：**通过词向量的线性运算（加减法），来捕捉语料库中的统计规律。**

我们可以分层级来拆解这些公式：

## 核心思想：Log-bilinear Model（对数双线性模型）

$$w_i \cdot w_j = \log P(i|j)$$

- **$w_i, w_j$**：分别是词 $i$ 和词 $j$ 的词向量。
    
- **$w_i \cdot w_j$**：两个向量的**点积**（内积）。在空间中，内积越大，表示两个向量越接近、越相似。
    
- **$P(i|j)$**：**条件概率**，表示在上下文词 $j$ 出现的情况下，词 $i$ 出现的概率。
    
- **意思**：GloVe 希望两个词的“向量相似度”去拟合它们在真实文本中的“共现概率的对数”。
    
## Vector Differences（向量差与比例）
$$w_x \cdot (w_a - w_b) = \log \frac{P(x|a)}{P(x|b)}$$

- **$w_a, w_b$**：两个基准词（比如 "冰" 和 "蒸汽"）。
    
- **$w_x$**：一个测试词（比如 "固体"）。
    
- **$w_a - w_b$**：**向量差**。这是 GloVe 的精髓，它代表了词 $a$ 和 $b$ 之间的语义差异。
    
- **$\frac{P(x|a)}{P(x|b)}$**：**共现概率的比率**。
    
    - 如果 $x$ 与 $a$ 相关但与 $b$ 无关（如 $x$ 是“固体”），这个比率会很大。
        
    - 如果 $x$ 与两者都相关或都无关，比率接近 1。
        
- **意思**：GloVe 认为，单一的概率 $P(x|a)$ 没法很好地区分词义，只有通过**概率的比率**才能精准捕捉语义区别。模型通过学习，使得向量空间的几何差异（减法）能对应上统计学上的概率比。
    
## 最终目标：Loss Function（损失函数）

为了让随机初始化的向量达到上面的理想状态，我们需要通过这个损失函数 $J$ 进行训练：

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

- **$V$**：词表大小。
    
- **$X_{ij}$**：词 $i$ 和词 $j$ 在整个语料库中共同出现的**次数**（这就是你提到的共现矩阵里的数值）。
    
- **$w_i^T \tilde{w}_j$**：词 $i$ 作为中心词和词 $j$ 作为上下文词时的向量内积。
    
- **$b_i, \tilde{b}_j$**：**偏置项（Bias）**。用来吸收掉一些由于词频本身高低带来的干扰（比如“的”、“了”这种词天生概率高，偏置项可以抵消这种影响）。
    
- **$\log X_{ij}$**：我们要拟合的目标。即希望内积加偏置后的结果，尽可能接近共现次数的对数。
    
- **$f(X_{ij})$**：**权重函数**（右下角的绿线图）。
    
    - 因为有些词共现次数太多（如 "of the"），如果不加权重，模型会被这些无意义的词统治。
        
    - $f(X_{ij})$ 的作用是：如果次数太少，权重低（减少噪声）；如果次数极多，权重封顶（右侧的平线），防止高频词权重过大。
        

**它具体在干嘛？**

1. **统计**：先扫描全语料库，生成一张巨大的**共现矩阵**（记录 $X_{ij}$）。
    
2. **初始化**：给每个词随机生成一个小向量 $w$。
    
3. **对齐**：通过梯度下降减少损失函数 $J$。不断微调 $w$ 的数值，直到每两个词向量的**内积**都能完美对应它们在语料库里的**共现规律**。
    
4. **结果**：训练完成后，你的词向量空间就具备了“线性语义”特性——不仅相似词靠得近，而且 $w_{man} - w_{woman} \approx w_{king} - w_{queen}$ 这种关系会自动浮现。
    

**它的优点**：

- **Fast training**：只扫一遍语料库计数，后续只针对非零矩阵元素训练，非常快。
    
- **Scalable**：可以处理极大的语料库。

## GloVe 可视化
![[Pasted image 20260516153404.png]]

# 如何评价词向量（Word Vectors）的好坏

在 NLP（自然语言处理）领域，评价方法通常分为两大类：**内部评价（Intrinsic）** 和 **外部评价（Extrinsic）**
## 内部评价 (Intrinsic Evaluation)

- 对特定或中间子任务的评估。
    
- 不把词向量放到复杂的实际应用（如翻译、搜索）中，而是直接测试向量本身的性质。
    
    - **计算快**：通常只需几秒钟。
        
    - **有助于理解系统**：能直观看到词向量是否抓住了语义关系。
        
    - **局限性**：除非已经证明该指标与最终任务高度相关，否则“内部评价”得高分并不代表最终产品就好用。
### 内部评价的具体案例——词向量类比

#### **核心逻辑：a : b :: c : ?**

翻译过来就是：**“a 之于 b，相当于 c 之于什么？”**

经典的例子是：**男人 : 女人 :: 国王 : 女王**

#### **公式解析：**

$$d = \arg \max_i \frac{(x_b - x_a + x_c)^T x_i}{\|x_b - x_a + x_c\|}$$

- **$x_b - x_a$**：代表一种“语义偏移”。比如（女人 - 男人）代表了“性别”这个维度的差值。
    
- **$x_b - x_a + x_c$**：就是在词 $c$（国王）的基础上，加上这个“性别偏移”。我们希望算出来的结果在向量空间中靠近 $d$（女王）。
    
- **$\arg \max$ 部分**：本质是在计算**余弦相似度**。在词表中搜索一个词 $x_i$，看哪个词和计算出来的理想目标最接近。

### 词义相似度 (Meaning Similarity)： 另一种内部词向量评价方法

**核心逻辑**：比较“词向量计算出的距离”与“人类主观判断的相似度”之间的相关性。
    
**数据集示例：WordSim353**
![[Pasted image 20260516153817.png|400]]
这是一个经典数据集，里面雇佣了很多人对两个词的相似度打分（0-10分）。
 **表格解读**：
- `tiger` 和 `cat`：人类觉得挺像（7.35分）。
-  `stock`（股票）和 `jaguar`（美洲豹）：完全没关系（0.92分）。
- **评价标准**：如果你的词向量计算出 `tiger` 和 `cat` 的余弦相似度很高，而 `stock` 和 `jaguar` 很低，且这个排序趋势与人类打分一致，那你的词向量就是优秀的。

#### 相关性评估 (Correlation Evaluation)：词向量距离与人类判断的相关性评估
![[Pasted image 20260516154524.png|600]]
这张表展示了不同模型在多个相似度数据集（WS353, MC, RG, SCWS, RW）上的表现。

- **表格参数说明**：
    
    - **Model**: 各种词向量模型（SVD 是传统降维，CBOW/SG 是 Word2Vec，GloVe 是本课重点）。
        
    - **Size**: 训练语料库的大小（6B 代表 60 亿词，42B 代表 420 亿词）。
        
- **关键结论**：
    
    1. **语料库越大，效果越好**：看 GloVe 从 6B 增加到 42B 时，WS353 的得分从 65.8 飙升到 75.9。
        
    2. **GloVe 表现强劲**：在同等语料（6B）下，GloVe 往往优于 Word2Vec（CBOW/SG）和传统的 SVD 方法。
        
    3. **加粗数字**：代表该数据集下的最佳表现，基本被 42B 语料下的 GloVe 包揽。
        
## 外部评价 (Extrinsic Evaluation)

- 在实际任务中评估。
    
- 把词向量作为下游系统（如情感分析、机器翻译）的一部分。
    
    - **耗时久**：需要训练整个大模型才能得出准确率。
        
    - **难以定位问题**：如果效果不好，不确定是词向量的问题，还是模型其他部分或它们之间协作的问题。
        
    - **终极准则**：如果你保持其他部分不变，只换了一套词向量，准确率提升了，那就是真的赢了。
**例如**：
![[Pasted image 20260516154943.png|500]]
- **核心任务：命名实体识别 (Named Entity Recognition, NER)**
    
    - **定义**：识别文本中的人名、组织名、地理位置等。
        
    - **例子**：从 "Chris Manning lives in Palo Alto" 中识别出 `Chris Manning` 是人（Person），`Palo Alto` 是地点（Location）。
        
- **为什么 NER 能评价词向量？**
    
    - 一个好的词向量应该让同类实体的向量聚在一起。如果模型知道 "Palo Alto" 和 "Tokyo" 都是地点，它做识别就会容易得多。
        
- **表格解读**：
    
    - **Discrete**: 离散表示（传统方法，不使用稠密词向量），作为基准线。
        
    - **GloVe 的表现**：在 Dev（开发集）、ACE 和 MUC7（标准测试集）上，GloVe 的准确率基本都是最高的（加粗部分）。
        
    - **Winning!**：这印证了之前说的——如果你换了一套词向量（换成 GloVe），下游任务（NER）的准确率提升了，那就证明这个词向量是真的好。

# 总结

总的来说，这一节还是蛮难的，不过在gimini的帮助下还是搞懂了大概！

在这节课程的学习中，主要掌握了以下核心内容：

- **词义表示的演变**：学习了从传统的 **WordNet**（基于人工定义的同义词库）和 **one-hot 编码**（离散且正交的向量）向现代**分布式语义学**的跨越。
    
- **分布语义学核心思想**：理解了“观其伴而知其义” ($You\ shall\ know\ a\ word\ by\ the\ company\ it\ keeps$)，即一个词的含义是由经常出现在其附近的上下文词决定的。
    
- **Word2vec 框架**：深入探讨了 **Skip-gram** 模型，学习了如何通过中心词预测上下文词，并利用 **Softmax** 函数将向量点积转化为概率分布。
    
- **训练与优化技巧**：掌握了如何通过负采样（Negative Sampling）解决原始 Softmax 计算量过大的问题，并理解了梯度下降（SGD）在更新稀疏词向量矩阵中的应用。
    
- **GloVe 模型**：学习了这种基于**全局词-词共现矩阵**的加权最小二乘模型，它结合了全局统计信息与词向量的线性子结构（如 $w_{king} - w_{man} + w_{woman} \approx w_{queen}$）。
    
- **评估与可视化**：了解了如何通过**内部评价**（词类比、词义相似度）和**外部评价**（如 NER 命名实体识别）来衡量词向量的质量，并学习了如何通过降维技术可视化词向量空间中的语义聚类。

挑战！阅读原论文：
1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper)
3. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper)

# Homework
[[HW1 Exploring Word Vectors]]
# NEXT
[[Lec4 Backpropagation and Neural Network Basics]]