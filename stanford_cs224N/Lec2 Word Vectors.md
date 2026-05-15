---
tags:
  - nlp
  - cs224N
date: 2026-05-14
---
# 阅读材料

[[cs224n-2026-lecture02-wordvecs.pdf]]
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

- **词语、短语等所代表的含义。** _(The idea that is represented by a word, phrase, etc.)_
    
- **一个人想要通过言语、符号等表达的想法。** _(The idea that a person wants to express by using words, signs, etc.)_
    
- **在文学作品、艺术作品等中所表达的主旨/意蕴。** _(The idea that is expressed in a work of writing, art, etc.)_
    
语言学中理解意义最常见的方式：

![[Pasted image 20260514143705.png]]

1. 核心公式：能指与所指

图片上方给出了一个等式：

**signifier (symbol) $\Leftrightarrow$ signified (idea or thing)**

- **能指 (Signifier / Symbol)：** 指的是语言的形式，比如单词的拼写（t-r-e-e）或发声。
    
- **所指 (Signified / Idea or thing)：** 指的是这个符号所代表的概念或具体的实物。
    
- **指称语义学 (Denotational Semantics)：** 这种观点认为，一个词的“意义”就在于它所**指代**（Denote）的那个对象。
    
2. 具体的例子

图片下方用“树”形象地展示了这一过程：

**tree $\Leftrightarrow$ {🌳, 🌲, 🌴, ...}**

- **左侧的 "tree"：** 是一个抽象的文本符号。
    
- **中间的 $\Leftrightarrow$：** 代表一种映射关系。
    
- **右侧的集合：** 代表现实世界中所有属于“树”范畴的实体。
    
**这意味着：**

在指称语义的视角下，单词 "tree" 的含义就是**所有现实中存在的树的集合**。当你使用这个词时，你就是在指向这个集合中的成员。

# 我们在计算机中如何拥有可用的意义？

## WordNet

此前最常用的 NLP（自然语言处理）解决方案：使用如 WordNet 之类的工具，这是一种包含同义词集（synonym sets）和上位词（hypernyms，即‘is a’关系）列表的词库。

![[Pasted image 20260514144616.png]]

## 像WordNet这样的资源存在的问题

- **资源虽有用但缺乏细微差别：**
    
    - **例如，“proficient”（精通）被列为“good”（好）的同义词。这仅在某些语境下是正确的。**
        
- **此外，WordNet 在某些同义词集中列出了冒犯性的词汇，却完全没有涵盖这些词的内涵或适用性。**
    
- **缺失词汇的新含义：**
    
    - **例如：wicked, badass, nifty, wizard, genius, ninja, bombest（注：这些词在俚语中常有“酷”、“厉害”等新义）。**
        
- **无法做到实时更新！**
    
- **具有主观性。**
    
- **需要人工劳动来创建和调整。**
    
- **无法用于准确计算词语相似度。**

## one-hot编码

在传统 NLP 中，我们将词视为离散符号：例如 hotel（酒店）、conference（会议）、motel（汽车旅馆）——这被称为**局部表示**（localist representation）。

此类词汇符号可以用 **one-hot 向量**来表示：

`motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]`

`hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]`

- **向量维度** = 词汇表中的单词总数（例如：500,000+）
    
- **含义**：向量中只有一个 1，其余全为 0。

## one-hot编码的问题

**示例：在网页搜索中，如果用户搜索“Seattle motel”（西雅图汽车旅馆），我们希望能够匹配包含“Seattle hotel”（西雅图酒店）的文档。**

- **`motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]`**
    
- **`hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]`**
    

**但这两个向量是正交的（orthogonal）。**

**对于 One-hot 向量而言，不存在天然的相似度概念！**

**解决方案：**

- **可以尝试依赖 WordNet 的同义词列表来获取相似度？**
    
- **但众所周知，这种方法效果很差：存在不完整性等问题。**
    
- **取而代之的是：学习将相似度直接编码进向量本身。**

## 通过上下文表示词语

- **分布语义学（Distributional semantics）：一个词的含义是由经常出现在其附近的词决定的。**
    
- **“道其友，知其人”（语出 J. R. Firth，1957: 11；原意为：通过一个词周围的词来了解这个词）。**
    
- **这是现代统计自然语言处理（NLP）中最成功的思想之一！**
    
- **当一个词 $w$ 出现在文本中时，它的“上下文”（context）是指出现在其附近（在固定大小的窗口范围内）的一组词。**
    
- **我们利用 $w$ 出现的多种上下文环境，来构建 $w$ 的表示形式。**

![[Pasted image 20260514154643.png]]
# Word vectors

我们将为每个单词构建一个**稠密向量**（dense vector），通过精心选择，使该向量与出现在相似上下文中的单词向量具有相似性；我们使用**向量点积**（标量积）来衡量这种相似度。

**注：** 词向量也被称为（词）嵌入（embeddings）或（神经）词表示。它们属于一种**分布式表示**（distributed representation）。

例如：(额这里没讲清楚怎么来的)
- `banking = [0.286, 0.792, −0.177, −0.107, 0.109, −0.542, 0.349, 0.271]`
    
- `monetary = [0.413, 0.582, −0.007, 0.247, 0.216, −0.718, 0.147, 0.051]`

## Word2vec

