# PCA

假设在 $\mathbb{R}^n$ 中有 $m$ 个点 $\{x^{(1)}, \cdots, x^{(m)}\}$, 我们希望对它们进行有损压缩 (使用更少的内存, 但损失一些精度去存储这些点) 投影到 $\mathbb{R}^p$ 维空间. 为了使得损失尽可能少, 我们考虑编码函数 $f$ 和解码函数 $g$:

$$
\begin{aligned}
f:&\mathbb{R}^n \rightarrow \mathbb{R}^k & g:&\mathbb{R}^p \rightarrow \mathbb{R}^n\\
&x \mapsto z &&z \mapsto Az
\end{aligned}
$$

其中, $A^TA = I_p$, 即 $A$ 的列向量是 $\mathbb{R}^p$ 的一组标准正交基.

PCA 满足**最近重构性**, 即令 $z^{(i)} = f(x^{(i)})$, 有:

$$
\displaystyle\min_{z^{(i)}} \sum_{i=1}^m ||Az^{(i)} -x^{(i)}||^2 = 
$$

该优化问题的解是 $z^{(i)} = A^Tx^{(i)}$

由于 $A$ 是未知的, 为了获得 $A$, 我们定义设计矩阵定义为 $X = (x^{(1)}, x^{(2)}, \cdots, x^{(m)})^T$, 便有 $(AA^Tx^{(1)}, AA^Tx^{(2)}, \cdots, AA^Tx^{(m)})^T = XAA^T$, 因而

$$
\displaystyle\arg\min_A \sum_{i=1}^m ||AA^Tx^{(i)} - x^{(i)}||^2 = \displaystyle\arg\min_A ||XAA^T-X||_{\text{F}}^2 = \displaystyle\arg\max_A ||XA||_{\text{F}}^2
$$

又 $||XA||_{\text{F}}^2 = \displaystyle\sum_{i=1}^m ||A^T x^{(i)}||^2$ 可知, PCA 满足**最大可分性**. 很容易知道 PCA 的最近重构性与最大可分性是等价的. 因而, PCA 任务便可转换为求解 $X^TXA = \lambda A$ 的优化问题. 其中 $\lambda$ 是拉格朗日乘子. 故而, PCA 问题可以转换为 $X^TX$ 的特征值分解问题. 下面便是 PCA 的算法:

1. 对 $X$ 进行标准化处理, 即 $\overline{X} = \frac{X - \mu}{\sigma}$, 其中

$$
\mu = \frac{1}{m}\displaystyle\sum_{i=1}^m x^{(i)}, \sigma = \frac{1}{m-1} \displaystyle\sum_{i=1}^m ||x^{(i)}||^2 = \frac{1}{m-1}X^TX
$$

2. 计算 $\overline{X}^T\overline{X}$, 并对其进行特征值分解.
3. 求取最大的 $p$ 个特征值, 及其对应的特征向量, 记作 $A_1,A_2, \cdots, A_p$.
4. 输出 $A = (A_1,A_2, \cdots, A_p)$ 和降维后的数据集 $A^Tx^{(i)}$.