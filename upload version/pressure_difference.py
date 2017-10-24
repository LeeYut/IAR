# -*- coding: utf-8 -*- 
#可以提前将数据做差分，然后再做preprocess。特别是需要多个features对齐的情形
#若是只打算用气压计做，可以先做preprocess再运行本程序就好了。
import pandas as pd
import numpy as np
import os
def pressure_difference(folder, file):
    df = pd.read_csv(folder + '/' + file)
    #需要修改这里的tag，时间轴
    time = df.time
    #需要修改这里的tag，气压数据
    pressure_values = df.p.values
    difference_value = [0]
    for i in range(1, len(pressure_values)):
        difference_value.append(pressure_values[i] - pressure_values[i-1])
    new_df = pd.DataFrame({'time':time, 'p':difference_value})
    #输出文件是bb.csv
    new_df.to_csv(os.path.join(folder,"b_diff.csv"), index = False)	

#批量处理
folder_pre = 'same_floor'
folder_list = os.listdir(folder_pre)

for i in folder_list:
    folder = folder_pre + '/' + i
    #指定folder后通过修改这个部分，决定要取出并进行处理的文件有哪些
    pressure_difference(folder_pre + '/' + i, 'p_new.csv')