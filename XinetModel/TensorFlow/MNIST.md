# MNIST 数字识别问题

- 在 Yann LeCun 教授的[网站](http://yann.lecun.com/exdb/mnist)中对 MNIST 数据集做出了详细的介绍。

MNIST 数据集是 NIST 数据集的子集，它包含 $60\,000$ 张图片作为训练集，$10\,000$ 张图片作为测试集。在 MNIST 数据集中每一张图片都代表了 $0\sim9$ 中的一个数字。图片的大小为 $28 \times 28$, 且数字都出现在图片的正中间。

## 载入数据集

```{.python .input  n=2}
import tables as tb
from sklearn.model_selection import train_test_split
import tensorflow as tf

h5 = tb.open_file('E:/xdata/X.h5')
mnist = h5.root.mnist

X_train, X_val, y_train, y_val = train_test_split(
    mnist.trainX[:], mnist.trainY[:], test_size=0.1, random_state=42)

X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float64)
X_val = X_val.reshape((X_val.shape[0], -1)).astype(np.float64)
X_test = mnist.testX[:].reshape((mnist.testX.nrows, -1)).astype(np.float64)

print("Training data size: ", X_train.shape[0])
print("Validating data size: ", X_val.shape[0])
print("Testing data size: ", X_test.shape[0])
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Training data size:  54000\nValidating data size:  6000\nTesting data size:  10000\n"
 }
]
```

## 设置输入和输出节点的个数,配置神经网络的参数

```{.python .input  n=3}
INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数       
                              
BATCH_SIZE = 100     # 每次 batch 打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        
MOVING_AVERAGE_DECAY = 0.99  
```

## 定义辅助函数来计算前向传播结果，使用ReLU做为激活函数

```{.python .input  n=4}
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
```

```{.python .input  n=6}
tf.layers.dense?
```

## 定义训练过程
