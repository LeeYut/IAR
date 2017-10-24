# -*- coding: utf-8 -*-   
import numpy as np
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.models import model_from_json
import os
    
#返回值是一个array,是(969,)的形式		
def read_file(folder_name,tag):
    file_list = os.listdir(folder_name)
    result = np.array([])
    for file in file_list:
        df = pd.read_csv(folder_name+ '/' + file)
        data = np.array(df[tag].values)
        result = np.concatenate((result, data), axis = 0)
    return result

#scaling的对象不能是单纯一维的，所以要reshape. [1,2,3,4] -> [[1],[2],[3],[4]],成为了一列，均值就是指的这一列的，是2.5
def scaling(result):
    result = result.reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(result)
    mean = scaler.mean_
    scale = scaler.scale_
    data = scaler.transform(result)
    return mean, scale, data

#输入是scaling之后的数据，我们这里将两个zip起来，注意zip只对[1,2,3],[4,5,6]工作得很好
def merge(data1, data2):
    merge = []
    data1 = data1.reshape(1, len(data1))
    data2 = data2.reshape(1, len(data2))
    print (data1.shape)
    z = zip(data1[0], data2[0])
    for i in list(z):
        merge.append(list(i))
    return merge

#这里将merge后的数据，整理成为带有time_step的数据
def time_transform(merge, time_step, type):
    merge_array = np.array(merge)
    x, y = [],[]
    #convert merge_array to array with time_step
    for i in range(len(merge) - time_step + 1):
        x.append(merge_array[i:i+time_step])
    if(type == 'esc_up'):
        y = np.array([1,0,0]*len(x))
    elif(type == 'esc_down'):
        y = np.array([0,1,0]*len(x))
    else:
        y = np.array([0,0,1]*len(x))

    X = np.array(x)
    #这里y要变形，这里的y就是一个纯的list，注意如果有三类数据就需要，reshape的第三维度是3
    Y = y.reshape(-1,3)
    return X, Y

def create_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences = False))
    #model.add(LSTM(100))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=2)
    return model
	
#文件夹的形式是 模式类别
#             /加速度数据文件夹，  /气压数据文件夹
#               /所有加速度文件  /所有气压文件
#注意存放的名字和顺序必须对应 
#acc_lift_down = read_file('data/lift_down_train/acc', 'acc')
#baro_lift_down = read_file('data/lift_down_train/baro', 'p')

#acc_walk = read_file('data/walk_train/acc', 'acc')
#baro_walk = read_file('data/walk_train/baro', 'p')

#acc_walk_up = read_file('data/upstairs/acc', 'acc')
#baro_walk_up = read_file('data/upstairs/baro', 'p')

#a = merge(acc_lift_down,baro_lift_down)
#b = merge(acc_walk,baro_walk)
#c = merge(acc_walk_up, baro_walk_up)
#只要我们选用的特征的维数不变，我们就只需要修改这里的需要连接的数量，三个类别就连接a,b,c
#a_b_c_merge = np.concatenate((a,b,c), axis = 0)
#以下部分用于对于全部的数据求出正确的均值和方差，用于测试组正规化数据

#scaler = preprocessing.StandardScaler()
#scaler.fit(a_b_c_merge)

#对数据组a b进行正规化
#a_norm = scaler.transform(a)
#b_norm = scaler.transform(b)
#c_norm = scaler.transform(c)
#a_norm = (a)
#b_norm =(b)
#c_norm = (c)

baro_esc_up = read_file('escalator_up/baro', 'p')
baro_esc_down = read_file('escalator_down/baro', 'p')
baro_same_floor = read_file('same_floor/baro', 'p')

#对a b分别做含有time step的变换,仅仅考虑单个数据值的情形
x1, y1 = time_transform(baro_esc_up, 40, 'esc_up')
x2, y2 = time_transform(baro_esc_down, 40, 'esc_down')
x3, y3 = time_transform(baro_same_floor, 40, 'same_floor')

x1 = x1.reshape(len(x1), len(x1[0]), 1)
x2 = x2.reshape(len(x2), len(x2[0]), 1)
x3 = x3.reshape(len(x3), len(x3[0]), 1)

#考虑多个feature的情形
#x1, y1 = time_transform(a_norm, 40, 'lift')
#x2, y2 = time_transform(b_norm, 40, 'walk')
#x3, y3 = time_transform(c_norm, 40, 'walk_up')
print("...............")
print (x1.shape)
print (x2.shape)
print(x3.shape)
print (y3.shape)
print(x1)
#合并并且打乱顺序
X = np.concatenate((x1,x2,x3), axis = 0)
Y = np.concatenate((y1, y2,y3), axis = 0)
#X = np.concatenate((x1,x2), axis = 0)
#Y = np.concatenate((y1, y2), axis = 0)
X, Y = shuffle(X, Y)
#数据已经准备完成，可以送去训练模型了
model = create_model(X, Y)
#保存模型
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print ('mean')
print (scaler.mean_)
print ('scale')
print (scaler.scale_)