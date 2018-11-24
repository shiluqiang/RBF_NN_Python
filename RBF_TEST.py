# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:49:44 2018

@author: lj
"""
from numpy import * 
from RBF_TRAIN import get_predict

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
    '''
    f = open(file_name)  # 打开文件
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))        
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    return mat(feature_data)

def generate_data():
    '''在[-4.5,4.5]之间随机生成20000组点
    '''
    # 1、随机生成数据点
    data = mat(zeros((20000, 2)))
    m = shape(data)[0]
    x = mat(random.rand(20000, 2))
    for i in range(m):
        data[i, 0] = x[i, 0] * 9 - 4.5
        data[i, 1] = x[i, 1] * 9 - 4.5
    # 2、将数据点保存到文件“test_data”中
    f = open("test_data.txt", "w")
    m,n = shape(data)
    for i in range(m):
        tmp =[]
        for j in range(n):
            tmp.append(str(data[i,j]))
        f.write("\t".join(tmp) + "\n")
    f.close()       

def load_model(file_center, file_delta, file_w):
    
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return mat(model)
    
    # 1、导入rbf函数中心
    center = get_model(file_center)
    
    # 2、导入rbf函数扩展常数
    delta = get_model(file_delta)
    
    # 3、导入隐含层到输出层之间的权重
    w = get_model(file_w)


    return center, delta, w

def save_predict(file_name, pre):
    '''保存最终的预测结果
    input:  pre(mat):最终的预测结果
    output:
    '''
    f = open(file_name, "w")
    m = shape(pre)[0]
    result = []
    for i in range(m):
        result.append(str(pre[i, 0]))
    f.write("\n".join(result))
    f.close()

if __name__ == "__main__":
    generate_data()
    # 1、导入测试数据
    print ("--------- 1.load data ------------")
    dataTest = load_data("test_data.txt")
    # 2、导入BP神经网络模型
    print ("--------- 2.load model ------------")
    center,delta,w = load_model("center.txt", "delta.txt", "weight.txt")
    # 3、得到最终的预测值
    print ("--------- 3.get prediction ------------")
    result = get_predict(dataTest, center, delta, w)
    # 4、保存最终的预测结果
    print ("--------- 4.save result ------------")
    save_predict("test_result.txt", result)
    
    
    
    
    
    
    
    
    
    

