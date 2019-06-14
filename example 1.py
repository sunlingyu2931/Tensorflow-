import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32) # create data
y_data = x_data * 0.1 +0.3 # 真实值

# create tensorflow stracture (start)
Weight = tf.Variable(tf.random_uniform([1],-1,1)) # weight 一般用variable inpout output 神经元等用constant
biases = tf.Variable(tf.zeros([1]))
# 和上面的y_data对比可以知道我们将weight define在-1-1之间而实际上上0。1，bias是0实际上是0。3。。通过不断的学习最后优化到0。1和0。3

y = Weight * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))  #cost function
optimizer = tf.train.GradientDescentOptimizer(0.5) #learnining rate 0-1
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()  #variable 需要进行初始化



sess = tf.Session()
sess.run(init) # 激活initial

for step in range(201):  # 训练201步
    sess.run(train)     # 指向train
    if step % 20 == 0:  # %取余，所以是每20步打印一次
        print(step,sess.run(Weight),sess.run(biases))  # 激活weight biases


#sess.run（）很重要，要注意