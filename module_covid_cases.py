# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:40:54 2022

@author: _K
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error


#%% Statics
MMS_PATH = os.path.join(os.getcwd(),'mms.pkl')


#%% Load the model
with open(MMS_PATH,'rb') as file:
    mms = pickle.load(file)


#%%

class EDA():
    
    def __init__(self):
        pass

    def plot_graph_train(self,df):
        
        plt.figure()
        plt.plot(df['cases_new'])
        plt.legend(['cases_new'])
        plt.show()
        
    def plot_graph_test(self,test_df):
        
        plt.figure()
        plt.plot(test_df['cases_new'])
        plt.legend(['cases_new'])
        plt.show()


class ModelCreation():
    def __init__(self):
        pass
    
    def model_development(self,X_train,num_node=64,
                          drop_rate=0.2,output_node=1):
        
        model=Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1))) #input_length 
        model.add(LSTM(num_node,return_sequences=True)) 
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node)) 
        model.add(Dropout(drop_rate))
        model.add(Dense(128))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node)) #output layer
        model.summary()
        
        return model


class PlotGraph():
    def __init__(self):
        pass
    
    def plot_result(self,test_df,predicted,mms):

        plt.figure()
        plt.plot(test_df,'b',label='actual covid cases')
        plt.plot(predicted,'r',label='predicted covid cases')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',
                 label='actual covid cases')
        plt.plot(mms.inverse_transform(predicted),'r',
                 label='predicted covid cases')
        plt.legend()
        plt.show()

class ErrorResults():
    def __init__(self):
        pass
    def final_results(self,test_df,predicted):
        
        print('mae:',mean_absolute_error(test_df,predicted))
        print('mse:',mean_squared_error(test_df,predicted))
        print('mape:',mean_absolute_percentage_error(test_df,predicted))
        
        test_df_inversed = mms.inverse_transform(test_df)
        predicted_inversed = mms.inverse_transform(predicted)
        
        print('mae_inversed:',mean_absolute_error(
            test_df_inversed,predicted_inversed))
        print('mse_inversed:',mean_squared_error(
            test_df_inversed,predicted_inversed))
        print('mape_inversed:',mean_absolute_percentage_error(
            test_df_inversed,predicted_inversed))
        
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    