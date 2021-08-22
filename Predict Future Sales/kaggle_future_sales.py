# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 18:02:45 2021

@author: r.muema
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def load_df():
    df_train = pd.read_csv('v_sales.csv')
    df_test = pd.read_csv('v_test.csv')
    df_submit = pd.read_csv('sample_submission.csv')
    # print(df_train.shape)
    # print(df_train.head())
    return df_train, df_test, df_submit

def pre_process(df):
    x = df.iloc[1:,:-1]
    y = df.iloc[1:,-1:]
    # model = RandomForestClassifier(n_estimators=200,max_depth=5,n_jobs=-1,random_state=48)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=48)
    return x_train, x_test, y_train, y_test

def build_rf_model(x_train, x_test, y_train, y_test, df_test, df_submit):
    model = RandomForestRegressor(n_estimators=200,max_depth=5,n_jobs=-1,random_state=48)
    model.fit(x_train,y_train.values.ravel())
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Random Forest Mean Squared Error - ', mse)
    test_pred = model.predict(df_test)
    df_submit['item_cnt_month'] = test_pred
    df_submit = df_submit.round(1)
    print(df_submit.head())
    # Save the dataframe to csv
    df_submit.to_csv('C:\\Users\\r.muema\\Documents\\Study\\Kaggle Competitions\\Predict Future Sales\\v_submit.csv', index=False, index_label=True)
        

def build_nn_model(x_train, x_test, y_train, y_test, df_test, df_submit):
    x_train = MinMaxScaler().fit_transform(x_train)
    x_test = MinMaxScaler().fit_transform(x_test)
    df_test = MinMaxScaler().fit_transform(df_test)
    
    hidden_neurons = 100
    # opt = SGD(learning_rate=0.9)
    opt = 'adam'
    
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    
    model.compile(loss='mean_squared_error', metrics=['mean_squared_error'], optimizer=opt)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, verbose=2)
    
     # Evaluate the model
    # _, train_acc = model.evaluate(x_train, y_train, verbose=1)
    # _, test_acc = model.evaluate(x_test, y_test, verbose=1)
    # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Neural Network Mean Squared Error - ', mse)
    test_pred = model.predict(df_test)
    df_submit['item_cnt_month'] = test_pred
    df_submit = df_submit.round(1)
    print(df_submit.head())
    # Save the dataframe to csv
    df_submit.to_csv('C:\\Users\\r.muema\\Documents\\Study\\Kaggle Competitions\\Predict Future Sales\\v_submit_nn.csv', index=False, index_label=True)
        
    
def main():
    df_train, df_test, df_submit = load_df()
    x_train, x_test, y_train, y_test = pre_process(df_train)
    # build_rf_model(x_train, x_test, y_train, y_test, df_test, df_submit)
    build_nn_model(x_train, x_test, y_train, y_test, df_test, df_submit)
    

if __name__ == '__main__':
    main()