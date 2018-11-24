# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:06:27 2018

@author: lj
"""
from numpy import *
from math import sqrt

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
            label_data(mat):标签
            n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label.append(int(lines[-1]))      
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    n_output = 1
    
    return mat(feature_data), mat(label).transpose(), n_output

def linear(x):
    '''Sigmoid函数（输出层神经元激活函数）
    input:  x(mat/float):自变量，可以是矩阵或者是任意实数
    output: Sigmoid值(mat/float):Sigmoid函数的值
    '''
    return x
   

def hidden_out(feature,center,delta):
    '''rbf函数（隐含层神经元输出函数）
    input:feature(mat):数据特征
          center(mat):rbf函数中心
          delta(mat)：rbf函数扩展常数
    output：hidden_output（mat）隐含层输出
    '''
    m,n = shape(feature)
    m1,n1 = shape(center)
    hidden_out = mat(zeros((m,m1)))
    for i in range(m):
        for j in range(m1):
            hidden_out[i,j] = exp(-1.0 * (feature[i,:]-center[j,:]) * (feature[i,:]-center[j,:]).T/(2*delta[0,j]*delta[0,j]))        
    return hidden_out


def predict_in(hidden_out, w):
    '''计算输出层的输入
    input:  hidden_out(mat):隐含层的输出
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: predict_in(mat):输出层的输入
    '''
    m = shape(hidden_out)[0]
    predict_in = hidden_out * w
    return predict_in

    
def predict_out(predict_in):
    '''输出层的输出
    input:  predict_in(mat):输出层的输入
    output: result(mat):输出层的输出
    '''
    result = linear(predict_in)
    return result

def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    '''计算隐含层的输入
    input:  feature(mat):特征
            label(mat):标签
            n_hidden(int):隐含层的节点个数
            maxCycle(int):最大的迭代次数
            alpha(float):学习率
            n_output(int):输出层的节点个数
    output: center(mat):rbf函数中心
            delta(mat):rbf函数扩展常数
            w(mat):隐含层到输出层之间的权重
    '''
    m, n = shape(feature)
    # 1、初始化
    center = mat(random.rand(n_hidden,n))
    center = center * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((n_hidden,n))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))
    delta = mat(random.rand(1,n_hidden))
    delta = delta * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - mat(ones((1,n_hidden))) * (4.0 * sqrt(6) / sqrt(n + n_hidden))   
    w = mat(random.rand(n_hidden, n_output))
    w = w * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - mat(ones((n_hidden, n_output))) * (4.0 * sqrt(6) / sqrt(n_hidden + n_output))

    
    # 2、训练
    iter = 0
    while iter <= maxCycle:
        # 2.1、信号正向传播
        # 2.1.1、计算隐含层的输出
        hidden_output = hidden_out(feature,center,delta)
        # 2.1.3、计算输出层的输入
        output_in = predict_in(hidden_output, w)  
        # 2.1.4、计算输出层的输出
        output_out = predict_out(output_in)
        
        # 2.2、误差的反向传播
        error = mat(label - output_out)
        for j in range(n_hidden):
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            for i in range(m):
                sum1 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j])) * (feature[i] - center[j])
                sum2 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j])) * (feature[i] - center[j]) * (feature[i] - center[j]).T
                sum3 += error[i,:] * exp(-1.0 * (feature[i]-center[j]) * (feature[i]-center[j]).T / (2 * delta[0,j]*delta[0,j]))
            delta_center = (w[j,:]/(delta[0,j]*delta[0,j])) * sum1
            delta_delta = (w[j,:]/(delta[0,j]*delta[0,j]*delta[0,j])) * sum2
            delta_w = sum3
       # 2.3、 修正权重和rbf函数中心和扩展常数       
            center[j,:] = center[j,:] + alpha * delta_center
            delta[0,j] = delta[0,j] + alpha * delta_delta
            w[j,:] = w[j,:] + alpha * delta_w
        if iter % 10 == 0:
            cost = (1.0/2) * get_cost(get_predict(feature, center, delta, w) - label)
            print ("\t-------- iter: ", iter, " ,cost: ",  cost)
        if cost < 3:   ###如果损失函数值小于3则停止迭
            break               
        iter += 1           
    return center, delta, w

def get_cost(cost):
    '''计算当前损失函数的值
    input:  cost(mat):预测值与标签之间的差
    output: cost_sum / m (double):损失函数的值
    '''
    m,n = shape(cost)
    
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i,j] * cost[i,j]
    return cost_sum / 2

def get_predict(feature, center, delta, w):
    '''计算最终的预测
    input:  feature(mat):特征
            w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: 预测值
    '''
    return predict_out(predict_in(hidden_out(feature,center,delta), w))

def save_model_result(center, delta, w, result):
    '''保存最终的模型
    input:  w0(mat):输入层到隐含层之间的权重
            b0(mat):输入层到隐含层之间的偏置
            w1(mat):隐含层到输出层之间的权重
            b1(mat):隐含层到输出层之间的偏置
    output: 
    '''
    def write_file(file_name, source):   
        f = open(file_name, "w")
        m, n = shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")
        f.close()
    
    write_file("center.txt", center)
    write_file("delta.txt", delta)
    write_file("weight.txt", w)
    write_file('train_result.txt',result)
    
def err_rate(label, pre):
    '''计算训练样本上的错误率
    input:  label(mat):训练样本的标签
            pre(mat):训练样本的预测值
    output: rate[0,0](float):错误率
    '''
    
    m = shape(label)[0]
    for j in range(m):
        if pre[j,0] > 0.5:
            pre[j,0] = 1.0
        else:
            pre[j,0] = 0.0

    err = 0.0
    for i in range(m):
        if float(label[i, 0]) != float(pre[i, 0]):
            err += 1
    rate = err / m
    return rate

if __name__ == "__main__":
    # 1、导入数据
    print ("--------- 1.load data ------------")
    feature, label, n_output = load_data("data.txt")
    # 2、训练网络模型
    print ("--------- 2.training ------------")
    center, delta, w = bp_train(feature, label, 20, 5000, 0.008, n_output)
    # 3、得到最终的预测结果
    print ("--------- 3.get prediction ------------")
    result = get_predict(feature, center, delta, w)    
    print ("训练准确性为：", (1 - err_rate(label, result)))
    # 4、保存最终的模型
    print ("--------- 4.save model and result ------------")
    save_model_result(center, delta, w, result)
