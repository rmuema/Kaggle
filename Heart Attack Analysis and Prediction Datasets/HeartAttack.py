# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:09:32 2021

@author: r.muema
"""
import yaml
import sys
try:
        with open ('C:/Users/r.muema/Documents/Study/Kaggle Competitions/Heart Attack Analysis and Prediction Datasets/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            utils = config['files']['utilities']
            sys.path.insert(1, utils)
except Exception as e:
        # print('Error : ' + e)
        print('Error : ' + str(e))
import pandas as pd
import numpy as np
import scipy.stats as st

import utilities as utils
import nn_classification_experiment as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from pickle import dump

def load_dataset(dataset):
    # Load dataset
    df = pd.read_csv(dataset)
    
    # Return dataset
    return df


def model_build(model, x_train, x_test, y_train, y_test):
    model.fit(x_train,y_train.values.ravel())
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
   
    
    # Calculate Accuracy - Train
    print(accuracy_score(y_train, y_pred_train), ' - RF Train Accuracy')
    
    # Calculate precision
    print(precision_score(y_train, y_pred_train), ' - RF Train Precision')
    
    # Calculate recall
    print(recall_score(y_train, y_pred_train), ' - RF Train Recall')
    
    # Calculate Accuracy - Test
    print(accuracy_score(y_test, y_pred_test), ' - RF Test Accuracy')
    
    # Calculate precision
    print(precision_score(y_test, y_pred_test), ' - RF Test Precision')
    
    # Calculate recall
    print(recall_score(y_test, y_pred_test), ' - RF Test Recall')
    
    # create confusion matrix
    matrix = confusion_matrix(y_test, y_pred_test)

    # create pandas dataframe
    class_names = ['Play_No', 'Play_Yes']
    dataframe_Confusion = pd.DataFrame(matrix, index=class_names, columns=class_names)
    
    # create heatmap
    sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('./confusion_matrix.png')
    plt.show()
    plt.close()
    plt.clf()

def build_dt_model(df):
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    # x_reduced = utils.pca(x)
    # x_reduced = utils.rfa_rf(x,y)
    # x = x_reduced
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
    
    model = DecisionTreeClassifier(max_depth=5,random_state=48)
    # model = DecisionTreeClassifier(random_state=48, max_depth=10)
    
    model_build(model, x_train, x_test, y_train, y_test)
    

    
def build_rf_model(df) :
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    # model = RandomForestClassifier(n_estimators=200,max_depth=5,n_jobs=-1,random_state=48)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
    # model.fit(x_train,y_train.values.ravel())
    model = utils.rf_grid(x_train,y_train.values.ravel())
    
    model_build(model, x_train, x_test, y_train, y_test)
    
    # Save model
    # save_model(model, 'model_rf.sav')
    

def xgboost(df):
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    model = xgb.XGBClassifier(use_label_encoder=False)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
   
    model_build(model, x_train, x_test, y_train, y_test)
    
    
def build_svm_model(df):
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    x_reduced = utils.rfa_rf(x,y)
    x = x_reduced
    # model = make_pipeline(StandardScaler(), SVC(random_state=48))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
    model = utils.svm_grid(x_train,y_train.values.ravel())
    
    model_build(model, x_train, x_test, y_train, y_test)

def build_nn_model(df):
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    # x_reduced = utils.pca(x)
    x_reduced = utils.rfa_rf(x,y)
    x = x_reduced
    x = MinMaxScaler().fit_transform(x)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
    
    hidden_neurons = 100
    # opt = SGD(learning_rate=0.9)
    opt = 'adam'
    
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
      
    model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=opt)
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, verbose=0)
    
    # Evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=1)
    _, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('val_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

def save_model(model, name):
    path = config['files']['model_output']
    file_path = path + name
    dump(model, open(file_path, 'wb'))
    
def main():
    try:
        with open ('C:/Users/r.muema/Documents/Study/Kaggle Competitions/Heart Attack Analysis and Prediction Datasets/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        # Load the dataset
        dataset = config['files']['location']
        df = load_dataset(dataset)
        
        # # analyse the database
        # utils.feature_selection_classification()
        
        # # Create Decision Tree Classifier
        # build_dt_model(df)
        
        # # Create Random Forest Classifier
        build_rf_model(df)
        
        # # Implement Neural Network
        # build_nn_model(df)
        
        # Implement XGboost
        # xgboost(df)
        
        # Implement SVM
        # build_svm_model(df)
        
        
        
               
    except Exception as e:
        # print('Error : ' + e)
        print('Error : ' + str(e))
        

if __name__ == '__main__':
    main()