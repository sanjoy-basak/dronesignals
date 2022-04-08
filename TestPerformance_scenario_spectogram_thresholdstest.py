#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:46:27 2020

@author: sanjoybasak
"""


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np

import h5py

from tensorflow.python.client import device_lib
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Dropout, Add, Dense, Activation
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
print(device_lib.list_local_devices())
from numpy import linalg as la

from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score



classes=["dx4e","dx6i","MTx","Nineeg","Parrot","q205","S500","tello","WiFi","wltoys"]
mods = classes
NFFT=256
maxlen=256

save_in_pdf=False

#threshold=0.5


#filename1 = 'testonly/GenSpecmixedsig2_9_9_unknownSG.h5' # load the dataset 

filename1 = 'GenDataRSNT_10_snrss2_NW.h5' # load the dataset 
h5f = h5py.File(filename1, 'r')



X_train_t = h5f['X_train']
Y_train_t = h5f['labels_RSNT_F_train']

train_idx_t = h5f['train_idx']
X_test_t = h5f['X_test']
Y_test_t = h5f['labels_RSNT_F_test']
test_idx_t = h5f['test_idx']
snr_test_t =h5f['SNR_test']
num_sig_presence_train_t=h5f['num_sig_presence_train']
num_sig_presence_test_t=h5f['num_sig_presence_test']


#X_train=np.array(X_train_t[()])
X_test=np.array(X_test_t[()])
#Y_train=np.array(Y_train_t[()])
Y_test=np.array(Y_test_t[()])

train_idx=np.array(train_idx_t[()])
test_idx=np.array(test_idx_t[()])
snr_test=np.array(snr_test_t[()])
num_sig_presence_train=np.array(num_sig_presence_train_t[()])
num_sig_presence_test=np.array(num_sig_presence_test_t[()])
h5f.close()



print("--"*10)
#print("Training data IQ:",X_train.shape)
#print("Training labels:",Y_train.shape)
print("Testing data IQ",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*10)


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

def getConfusionMatrixPlot(true_labels, predicted_labels):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = mods 
    plt.xticks(range(width), alphabet[:width], rotation=30,fontsize=6)
    plt.yticks(range(height), alphabet[:height],fontsize=6)
    return plt



# initialize and train model
filepath="spectrogramweights1NW.hdf5"

filename1= 'resultshf/RSNT_results_RN1.h5'

model = load_model(filepath)
model.summary()

idddd=1

def makeplot(Y_true,Y_pred):
    Y_pred[Y_pred>threshold]=1
    Y_pred[Y_pred<threshold]=0
    acc_score = accuracy_score(Y_true,Y_pred)
    rec_score = recall_score(Y_true,Y_pred,average='micro')
    f1_score_cur = f1_score(Y_true,Y_pred,average='micro')
        
    print("acc score: ",acc_score," rec score: ",rec_score," f1 score: ",f1_score_cur)
    
    Ytrue_barplot=np.sum(Y_true,0)/Y_true.shape[0]
    Ypred_barplot=np.sum(Y_pred,0)/Y_pred.shape[0]
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    cls_num=np.arange(1,len(mods)+1,1)
    ax.grid()
    ax.bar(cls_num,Ytrue_barplot,width=0.8,color='r', align='center')
    ax.bar(cls_num,Ypred_barplot,width=0.6,color='b', align='center')
    
    accstr=f'{acc_score:.4f}'
    recstr=f'{rec_score:.4f}'
    f1str=f'{f1_score_cur:.4f}'
    title_temp='acc:'+accstr+',rec:'+recstr+',f1:'+f1str
    ax.set_title(title_temp)
    
    global idddd
    idddd=idddd+1
    filename="fig"+str(idddd)+".png"
    
    fig.savefig(filename)
    
    return fig
    


if save_in_pdf:
    pp = PdfPages('Multilabel_spectogramresult.pdf')



def compute_confmatrix_snr_getf1(snr,threshold):
    

    #lbl_idxss=np.where((np.array(snr_test)==snr) & (np.array(num_sig_presence_test)==7))
    lbl_idxss=np.where((np.array(snr_test)==snr))
    lbl_idxss=np.array(lbl_idxss)
    testdata_length=len(lbl_idxss[0])
    
    print("Length cur lbl: ",testdata_length)
    X_test2=np.zeros([testdata_length,X_test.shape[1],X_test.shape[2],1])
    Y_test2=np.zeros([testdata_length,len(mods)])
    
    
    index=0
    for ii in lbl_idxss[0,:]:
        X_test2[index,:,:,:]=X_test[ii,:,:,:]
        Y_test2[index,:]=Y_test[ii,:]
        index=index+1
        
    
    print("X_test2 shape: ",X_test2.shape)
    print("Y_test2 shape: ",Y_test2.shape)
    
    Y_true = Y_test2
    Y_pred = model.predict(X_test2,batch_size=32)
    ev_result=model.evaluate(X_test2,Y_true,batch_size=32)
    
    acc_score=ev_result[1]
    print('acc_score:',acc_score) 

    
    Y_true=np.argmax(Y_true,1)
    Y_pred=np.argmax(Y_pred,1)
	
    Y_true=Y_true.reshape(Y_true.shape[0])   
    Y_pred=Y_pred.reshape(Y_pred.shape[0]) 

    acc_score = accuracy_score(Y_true,Y_pred)
    #rec_score = recall_score(Y_true,Y_pred,average='micro')
    #f1_score_cur = f1_score(Y_true,Y_pred,average='micro')
    #precision_score_cur =precision_score(Y_true,Y_pred)
    rec_score = recall_score(Y_true,Y_pred, average='micro')
    f1_score_cur = f1_score(Y_true,Y_pred, average='micro')
    precision_score_cur =precision_score(Y_true,Y_pred, average='micro')    
    
   
    
    return acc_score,precision_score_cur,rec_score,f1_score_cur

def calculate_precision_recall(test_Y_i,test_Y_i_hat):
    true_labels=np.argmax(test_Y_i, 1)
    predicted_labels=np.argmax(test_Y_i_hat, 1)
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    precision=np.zeros((len(classes),1))
    recall=np.zeros((len(classes),1))
    acc_permodel=np.zeros((len(classes),1))
    for ii in range(len(classes)):
        TP_cur=cm[ii,ii]
        acc_permodel[ii]=TP_cur
        TP_FP_sum=np.sum(cm[:,ii])
        TP_FN_sum=np.sum(cm[ii,:])
        precision[ii]=TP_cur/TP_FP_sum
        recall[ii]=TP_cur/TP_FN_sum
    
    averageprecision=np.mean(precision)
    averagerecall=np.mean(recall)
    F1_score_model=2*(averageprecision*averagerecall)/(averageprecision+averagerecall)
    return acc_permodel,precision,recall,F1_score_model
    



"""
for ii in scenario_all:
    scenario_cur=ii
    compute_confmatrix_labelbased(ii)
    
"""

snrs=np.unique(snr_test)



thresholds=[0.5]

idx=0


accuracy_all=np.zeros((len(snrs)*len(thresholds),1))
precision_all=np.zeros((len(snrs)*len(thresholds),1))
recall_all=np.zeros((len(snrs)*len(thresholds),1))
f1_all=np.zeros((len(snrs)*len(thresholds),1))
threshold_all=np.zeros((len(snrs)*len(thresholds),1))
snr_all=np.zeros((len(snrs)*len(thresholds),1))


for ii in snrs:
    for threshold in thresholds:
        acc_cur,pres_cur,rec_cur,f1_cur=compute_confmatrix_snr_getf1(ii,threshold)
        accuracy_all[idx]=acc_cur
        precision_all[idx]=pres_cur
        recall_all[idx]=rec_cur
        f1_all[idx]=f1_cur
        #threshold_all[idx]=threshold
        snr_all[idx]=ii
        
        print('snr:',ii,",acc:",acc_cur,",precision score: ",pres_cur," rec score: ",rec_cur," f1 score: ",f1_cur)
        idx=idx+1



#h5f1=h5py.File(filename1, 'w')
#h5f1.create_dataset('accuracy', data=accuracy_all)
#h5f1.close()

h5f1=h5py.File(filename1, 'w')
h5f1.create_dataset('accuracy', data=accuracy_all)
h5f1.create_dataset('precision', data=precision_all)
h5f1.create_dataset('recall', data=recall_all)
h5f1.create_dataset('f1score', data=f1_all)
h5f1.create_dataset('snr', data=snr_all)
h5f1.close()


'''
filename1= 'RSNT_results_spectrogram.h5'
h5f1=h5py.File(filename1, 'w')
h5f1.create_dataset('accuracy', data=accuracy_all)
h5f1.create_dataset('precision', data=precision_all)
h5f1.create_dataset('recall', data=recall_all)
h5f1.create_dataset('f1score', data=f1_all)
h5f1.create_dataset('snr', data=snr_all)
h5f1.create_dataset('thresholds', data=threshold_all)
h5f1.close()
'''

if save_in_pdf:
    pp.close()   
