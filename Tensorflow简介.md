# Tensorflow简介

人工智能学习系统，用来完成深度学习

google开发的外部结构包，可以理解为python中的一个包



支持python，c++

支持rnn，lstm

可以被用来语音识别，图像处理等多项深度学习领域



可以在一个或多个cpu/gpu中运行

可以运行在嵌入性系统中 i.e. 手机，平板，pc，分布式系统中





## 进入主题啦 ^_^



Tensorflow是一个编程系统，

使用  图(graphs)  来表示计算任务

在 会话(session) 中执行 graphs

Graph中的节点称为operation，一个op获得0个或多个tensor，执行计算，再次产生tensor

![img](file:////Users/sunlingyu/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image001.png)

 

一个tensor和一个variable放在op这里，op相当于一个运算 (加减乘除等)

然后得到一个新的tensor

再进行运算等

 



Tensor表示数据

Variable也是表示数据

一般把神经网络一些input和一些中间值，output  define为tensor

一般把权值之类的define为variable，权值需要不断的调整和优化









```
#2种session方法
matrix1 = tf.constant([[3,3]])  # 1行2列
matrix2 = tf.constant([[2],[2]])  # 2行1列
product = tf.matmul(matrix1,matrix2)
# np.dot(m1,m2) numpy是这样

#method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)  # 不需要close
```



变量的使用方法：

```
#define variable
state = tf.Variable(0,name = 'counter') # variable
# print(state.name)
one = tf.constant(1) # tensor

new_value = tf.add(state,one)  # operation
update = tf.assign(state,new_value)  # 把state的值变为new_value

init = tf.initialize_all_variables() #initialize variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```



placehoder的使用方法：

```
import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict = {input1:7,input2:2}))  # placeholder 就是之后再传数据进去
```



```
# 线性tensorflow的运用
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100)  # 生成0-1之间的数字  randn 是标准正态分布的，均值为0标准差为1（有正有负的）
noise = np.random.normal(0,0.01,x_data.shape)
y_date = x_data * 0.1 + 0.2 + noise

# plt.scatter(x_data,y_date)
# plt.show()

# define model 用来算预测值与实际值之间的loss
d = tf.Variable(np.random.rand(1))
k = tf.Variable(np.random.rand(1))
y = k * x_data + d

loss = tf.losses.mean_squared_error(y_date,y)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(optimizer)
        if i % 20 ==0:
            print(i,sess.run([k,d]))
    y_pred = sess.run(y)
    plt.scatter(x_data,y_date)
    plt.plot(x_data,y_pred,'r-',lw = 3)
    plt.show()
```
