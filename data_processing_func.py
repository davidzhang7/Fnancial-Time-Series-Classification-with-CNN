"""
Created on Wed Dec 19 22:21:17 2018

@author: Dawei Zhang (UNI: dz2363) 
"""

"""All functions needed to create samples for training
1. class CNNSamples: a general class to store relevant information of a CNN sample, containing:
    a) some members recording information of the sample;
    b) a member function getTimeSeriesCNNSample() which creates specified type image data from given stock data frame.

2. getTimeSeriesCNNTarget(): obtain target variable from a given stock data frame. Variables are defined in the report.
3. getTrainingDataFromPath(): load and partition data into training, validating and testing.

Remarks: it is suggested that everything is left unchanged. Make this file visible to all the notebook files associated with the project,
         which will constantly call functions defined here.
                    
"""

import numpy as np
import pandas as pd
import pickle
import time
import random
from sklearn.metrics import confusion_matrix
from pykalman import KalmanFilter
from pyts.image import GASF, GADF, MTF

# Add this class from 'get training and testing samples.ipynb'
# Otherwise, will not be able to load in data (pickles of lists of instances of CNNSamples) previously saved
image_size = 16
channel_size = 4
numOfClass = 3

class CNNSamples:
    
    def __init__(self, 
                 method='GADF', 
                 data_len=64, 
                 image_size=16, 
                 retrain_freq=5):
        
        self.data_len = data_len
        self.image_size = image_size
        self.retrain_freq = retrain_freq
        self.GADFSample, self.GASFSample = [], []
        self.nSamples = 0
        self.method = method
        
    def getTimeSeriesCNNSample(self, stockData, useSmoothedData=False):
        
        self.permno = stockData.PERMNO.iloc[0]
        data = stockData.drop('PERMNO', axis=1).T
        self.nDays = data.shape[1]
        
        for i in range(self.data_len, self.nDays, self.retrain_freq):
            series = data.iloc[:, i-self.data_len:i]
            
            if useSmoothedData:
                Smoother = KalmanFilter(n_dim_obs=series.shape[0], n_dim_state=series.shape[0],
                                        em_vars=['transition_matrices','observation_matrices',
                                                 'transition_offsets','observation_offsets',
                                                 'transition_covariance','observation_convariance',
                                                 'initial_state_mean','initial_state_covariance'])
                measurements = series.T.values
                Smoother.em(measurements, n_iter=5)
                series, _ = Smoother.smooth(measurements)
                series = series.T
                
            if self.method == 'GADF':
                gadf = GADF(self.image_size)
                self.GADFSample.append(gadf.fit_transform(series).T)
                
            elif self.method=='GASF':
                gasf = GASF(self.image_size)
                self.GASFSample.append(gasf.fit_transform(series).T)
            
        self.nSamples = len(self.GADFSample)

        return self



def getTimeSeriesCNNTarget(df, data_len=64, retrain_freq=5, up_return = 0.0125, down_return = -0.0125):
    data = df[['PRC','VOL']]
    nSample = len(data)
    Targets = []
    
    for i in range(data_len, nSample, retrain_freq):
        if data.VOL.iloc[i-retrain_freq:i].values.sum()>0:
            lastVWAP = np.average(data.PRC.iloc[i-retrain_freq:i].values, 
                                  weights = data.VOL.iloc[i-retrain_freq:i].values)
        else:
            lastVWAP = data.PRC.iloc[i]
        if data.VOL.iloc[i:np.min([nSample-1, i+retrain_freq])].values.sum()>0:
            nextVWAP = np.average(data.PRC.iloc[i:np.min([nSample-1, i+retrain_freq])].values, 
                                  weights = data.VOL.iloc[i:np.min([nSample-1, i+retrain_freq])].values)
        else:
            nextVWAP = data.PRC.iloc[np.min([nSample-1, i+retrain_freq])]
            
        VWAPReturn = (nextVWAP - lastVWAP)/lastVWAP
        
        _long = [1,0,0]
        _hold = [0,1,0]
        _short = [0,0,1]
        
        if VWAPReturn > up_return:
            Targets.append(_long)
        elif VWAPReturn < down_return:
            Targets.append(_short)
        else:
            Targets.append(_hold)
            
    return (df.PERMNO.iloc[0], Targets)




def getTrainingDataFromPath(feature_path = 'data/cnn samples',
                            target_path = 'data/cnn samples',
                            data_type = 'GADF',
                            image_size = 16,
                            train_val_size = 2/3,
                            train_size = 0.75):
    
    """ This function helps load in all TS-Image samples, and organize them into trainable manner. 
        Parameters explained:
        
        1) feature_path, target_path:
            Path for saved data. 'feature_path': X data; 'target_path': Y data.
            Notice: X data has to be named as 'CNNSamples_1', 'CNNSamples_2', ... etc.
                    Y data has to be named as 'CNNSamples_target'
                    
        2) data_type:
            Takes value only from one of 'GADF' and 'GASF'. Specify the type of TS-Image type.
        
        3) image_size:
            Image_size of each sample.
            
        4) train_val_size:
            Portion of sample taken to do training+validation
        
        5) train_size:
            Portion of sample taken from training+validation data to do training
            
    """
    
    data_file_num = 27
    all_data = []
    for i in range(data_file_num):
        with open(feature_path+'CNNSamples_'+str(i+1), 'rb') as pick:
            all_data += pickle.load(pick)
            pick.close()
    
    X_train, X_val, X_test = [], [], []
    if data_type == 'GADF':
        for obj in all_data:
            X_train += (obj.GADFSample[:int((int(obj.nSamples*train_val_size)+1)*train_size)])
            X_val += (obj.GADFSample[int((int(obj.nSamples*train_val_size)+1)*train_size):(int(obj.nSamples*train_val_size)+1)])
            X_test += obj.GADFSample[int(obj.nSamples*train_val_size)+1:]
    else:
        for obj in all_data:
            X_train += (obj.GASFSample[:int((int(obj.nSamples*train_val_size)+1)*train_size)])
            X_val += (obj.GASFSample[int((int(obj.nSamples*train_val_size)+1)*train_size):(int(obj.nSamples*train_val_size)+1)])
            X_test += obj.GASFSample[int(obj.nSamples*train_val_size)+1:]    
            
    X_train = np.array(X_train).reshape((len(X_train), image_size, image_size, 4))
    X_val = np.array(X_val).reshape((len(X_val), image_size, image_size, 4))
    X_test = np.array(X_test).reshape((len(X_test), image_size, image_size, 4))
    
    with open(target_path+'CNNSamples_target', 'rb') as pick2:
        Ytmp = pickle.load(pick2)
        pick2.close()
    
    Y_train = np.array([y[1][:int((int(obj.nSamples*train_val_size)+1)*train_size)] for y in Ytmp]).reshape((len(X_train), 3))
    Y_val = np.array([y[1][int((int(obj.nSamples*train_val_size)+1)*train_size):(int(obj.nSamples*train_val_size)+1)] for y in Ytmp]).reshape((len(X_val), 3))
    Y_test = np.array([y[1][int(obj.nSamples*train_val_size)+1:] for y in Ytmp]).reshape((len(X_test), 3))
    
    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)




