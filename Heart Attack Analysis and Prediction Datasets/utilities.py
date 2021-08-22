# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 07:31:42 2021

@author: r.muema
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import scipy.stats as st

def plot_multiclass_roc(n_classes, y_test, y_score):
    '''
    https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

    Parameters
    ----------
    n_classes : TYPE
        Number of classes in the classifier.
    y_test : TYPE
        True class.
    y_score : TYPE
        Prediction results.

    Returns
    -------
    None.

    '''
    # Plot linewidth.
    lw = 2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.6)
    plt.ylim(0.6, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Zoom')
    plt.legend(loc="lower right")
    plt.show()

def CI(x):
   neg, plus = st.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=st.sem(x)) 
   # print(np.mean(x))
   # print(neg, ' ', plus)
   print(round(np.mean(x),4) , '±' , round(plus - np.mean(x), 4))
   return round(np.mean(x),4) , round(plus - np.mean(x), 4)

def convert_list(strlist):
    x = []
    strlist = str(strlist).replace(' ',',')
    for val in strlist.split(','):
        val = val.strip()
        if val:
            # print(val)
            x.append(float(val))
    return x

def convert_time_series(df,  a, b, output):
    '''
    Utility to convert a one column dataframe to a a x n file.
    Use for time series data

    Parameters
    ----------
    df : dataframe
        Dataframe with one column.
    a : INT
        Number of predictors required .
    b : INT
        The time lag (skip every b).
    output : FILENAME
        Create output in the filename specified.

    Returns
    -------
    None.

    '''
    # Create output file
    with open(output, 'w') as f:
        i = 0
        while i < len(df):
            if (i+a < len(df)):
                k = 0
                line = ''
                while k < a+1:
                    if k == a:
                        line = line + str(df.iloc[i+k][0])
                    else:
                        line = line + str(df.iloc[i+k][0]) + ','
                    k = k + 1
                # print (line, ' - line')
                print(line)
                f.write(line + '\n')
            i = i + b

def analyse_dataset(df):
    # Start with descriptive statistics
    
    # Get the dimensions
    print (df.shape, ' - dimensions')

    # Increase pandas display width to get a view of all columns
    pd.set_option('display.max_columns', 25)
    
    # Have a peek at the data
    print(df.head(), ' - sneak preview')
    
    # Look at the data attributes (object means string)
    print(df.dtypes, ' - data types')
    
    # Get count, mean, standard deviation of continous data
    print(df.describe(), ' - continous data statistics')
    
    # # Look at the class distributions for discrete values
    # discrete_vals = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    #                  'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 
    #                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    # for val in discrete_vals:
    #     print(df.groupby(val).size(), ' - ', val , ' class distribution')
    
    # Look at correlations of continous data to see if we find anything useful (Total charges and tenure have strongest relationship)
    print(df.corr(), ' - correlation')
    sns.heatmap(df.corr(), annot=True)
    
    # Draw some diagnostic plots
    
    # Pairplot of discrete values
    plt.clf()
    sns.pairplot(data=df)
   
    # Histogram
    print(df.hist())
    # Density plot
    df.plot(kind='density', subplots=True, layout=(df.shape[1],df.shape[1]))
    # Box plots
    df.plot(kind='box', subplots=True, layout=(df.shape[1],df.shape[1]))
   
    # Scatter plot
    plt.clf()
    sns.scatterplot(data=df)