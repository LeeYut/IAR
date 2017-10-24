# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np
from scipy import stats
def slope(file):
    df = pd.read_csv(file)
    data = df['P'].values
    #通过修改这里，来确认求取多长的一个slope
    x = np.linspace(0,4,40)
    for i in range(len(data)-40):
        y = data[i:i+40]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        print (slope)
folder = 'data/upstairs/baro'		
slope(folder  + "baro.csv")