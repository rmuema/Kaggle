# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:55:32 2021

@author: r.muema
"""
import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve 
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from utilities import plot_multiclass_roc



def assign_class(x):
    '''
    Function to assign classes to the Ring Age

    Parameters
    ----------
    x : int
        The original ring age.

    Returns
    -------
    int
        The class the ring age belongs to.

    '''
    if (x <= 7):
        return 1
    elif (7 < x <=10):
        return 2
    elif (10 < x <= 15):
        return 3
    elif (x > 15):
        return 4

def initial_load_cleanup():
    '''
    Initial Load and CleanUP of Abalone Data
    Sex after label encoder transformation and MaxMinScaler transformation
    Male --> 2 --> 1
    Infant --> 1 --> 0.5
    Female --> 0 --> 0
    
    Ring age classified into:
        Class 1, 2, 3 and 4

    Returns
    -------
    None.

    '''
    # Create column names for the dataset
    header = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    # Load original dataset
    abalone = pd.read_csv('data/abalone.data', names=header)
    # Start cleanup of data
    # Sex / nominal / -- / M, F, and I (infant) - encode to digits
    labelenc = LabelEncoder()
    abalone['Sex'] = labelenc.fit_transform(abalone['Sex'])
    # Rings / integer / -- / gives the age in years - this needs to be converted to classes for the classification problem
    abalone['Rings'] = abalone['Rings'].apply(lambda v: assign_class(v))
    # One hot encode the classes generated
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(abalone[['Rings']]).toarray())
    # Create column names for the classes
    enc_df.columns = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    # Join the encoded classes to the original dataset
    abalone = abalone.join(enc_df)
    # Drop the Rings column so that we can use the encoded columns
    abalone.drop(columns=['Rings'], inplace=True)
    # Get the columns names
    names = abalone.columns
    # Standardise the data
    abalone = pd.DataFrame(MinMaxScaler().fit_transform(abalone))
    # Insert the column names
    abalone.columns = names
    # Remove Height Outliers
    abalone_filtered = abalone[abalone['Height'] > 0]
    abalone_filtered = abalone_filtered[abalone_filtered['Height'] < 0.5]
    # Save the cleaned up 
    abalone_filtered.to_csv('classificationdata.csv', index=False)
    
def data_viz():
    # Create column names for the dataset
    header = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    # Load original dataset
    abalone = pd.read_csv('data/abalone.data', names=header)
    # Data visualisation
    
    # Create scatter plots of continous data with respect to Rings (target)
    # Length
    plt.figure()
    sns.scatterplot(x='Rings', y='Length', hue='Sex', data=abalone)
    plt.title('Length vs Rings', size=20)
    
    # Diameter
    plt.figure()
    sns.scatterplot(x='Rings', y='Diameter', hue='Sex', data=abalone)
    plt.title('Diameter vs Rings', size=20)
    
    # Whole weight
    plt.figure()
    sns.scatterplot(x='Rings', y='Whole weight', hue='Sex', data=abalone)
    plt.title('Whole weight vs Rings', size=20)
    
    # Shucked weight
    plt.figure()
    sns.scatterplot(x='Rings', y='Shucked weight', hue='Sex', data=abalone)
    plt.title('Shucked weight vs Rings', size=20)
    
    # Viscera weight
    plt.figure()
    sns.scatterplot(x='Rings', y='Viscera weight', hue='Sex', data=abalone)
    plt.title('Viscera weight vs Rings', size=20)
    
    # Shell weight
    plt.figure()
    sns.scatterplot(x='Rings', y='Shell weight', hue='Sex', data=abalone)
    plt.title('Shell weight vs Rings', size=20)
    
    # Height
    plt.figure()
    sns.scatterplot(x='Rings', y='Height', hue='Sex', data=abalone)
    plt.title('Height vs Rings', size=20)
    
    abalone_filtered = abalone[abalone['Height'] > 0]
    abalone_filtered = abalone_filtered[abalone_filtered['Height'] < 0.5]
    
    # Height filtered
    plt.figure()
    sns.scatterplot(x='Rings', y='Height', hue='Sex', data=abalone_filtered)
    plt.title('Height filtered vs Rings', size=20)
    
    # Create a pairplot of all features 
    sns.pairplot(abalone, hue='Sex')
    plt.figure()
    # Get correlation matrix
    corr = abalone.corr()
    # https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    )
    plt.title("Abalone Correlation Matrix", size=24)
    plt.figure()
    # Create a box plot of Sex and Rings feature
    sns.boxplot(x='Sex', y='Rings', data=abalone)
    plt.figure()
    # Create a histogram for original Rings
    sns.histplot(data = abalone, x='Rings')
    plt.figure()
    # Transform Rings to classes and then get histogram
    abalone['Rings'] = abalone['Rings'].apply(lambda v: assign_class(v))
    sns.histplot(data = abalone, x='Rings')
    plt.title("Abalone Age Distribution", size=24)
    plt.figure()
    # Create a pairplot of all features 
    sns.pairplot(abalone, hue='Rings')
    plt.figure()
    
def split_dataset(df, target):
    '''
    Provide single split dataset for the experiments

    Returns
    -------
    x_train : TYPE
        X Train Data.
    x_test : TYPE
        X Test Data.
    y_train : TYPE
        Y Train Data.
    y_test : TYPE
        Y Test Data.

    '''

    x_data = df.iloc[:,:target]
    y_data = df.iloc[:,target:]
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.4, random_state=48)
    
    
    
    return x_train, x_test, y_train, y_test

def epochs_experiment(x_train, x_test, y_train, y_test, MaxRun):
    '''
    Investigate the best number of epochs to be used for the dataset

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    MaxRun : TYPE
        DESCRIPTION.

    Returns
    -------
    INT
        Optimal number of epochs.

    '''
    # Define return values
    trainACC =  np.zeros(MaxRun)
    testACC =  np.zeros(MaxRun)
    epochs = [500,1000,2000,4000]
    # hidden_neurons = [5,10]
    train_best_acc =  np.zeros(len(epochs))
    test_best_acc =  np.zeros(len(epochs))
    
    # Write output to file
    with open('experiment_epochs.log', 'w') as f:
        f.write('Log for Experiment Number 1 - Hidden Neurons' + '\n\n\n')
        # Fit model using number of experimental runs
        # i = 0
        for index, epoch in enumerate(epochs):
        # while i < len(hidden_neurons):
            for run in range(0, MaxRun  ):
                
                print('experimental run number ' , run , ' with ' , epoch , 'number of epochs') 
                # Define model
                model = Sequential()
                model.add(Dense(200, input_dim=x_train.shape[1], activation='relu'))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))
                model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
                # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
                
                # Evaluate the model
                _, train_acc = model.evaluate(x_train, y_train, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
                trainACC[run] = train_acc
                testACC[run] = test_acc
            
            print(' print classification performance for each experimental' ) 
            print(trainACC, ' - Train')
            print(testACC, ' - Test')
            
            print(' print mean and std of training performance') 
            print(np.mean(trainACC), np.std(trainACC), ' - Train')
            # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            print(st.t.interval(0.95, len(trainACC)-1, loc=np.mean(trainACC), scale=st.sem(trainACC)), ' - 95% CI Train')
            print(np.mean(testACC), np.std(testACC),' - Test')
            print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Test')
            
            # Write to file
            f.write('experimental results with '+ str(epoch) + ' number of epochs' + '\n\n')
            f.write(' print classification performance for each experimental' + '\n' ) 
            f.write(str(trainACC) + ' - Train' + '\n')
            f.write(str(testACC) + ' - Test' + '\n')
            
            f.write(' print mean and std of training performance' + '\n') 
            f.write(str(np.mean(trainACC)) + str(np.std(trainACC)) + ' - Train' + '\n')
            f.write(str(np.mean(testACC)) + str(np.std(testACC)) +' - Test' + '\n\n')
            
            # Save results 
            train_best_acc[index] = np.mean(trainACC)
            test_best_acc[index] = np.mean(testACC)
        # Print results to screen
        print('Best Results')
        print(train_best_acc, ' - Train')
        print(np.max(train_best_acc), ' - Best Train')
        print(test_best_acc, ' - Test')
        print(np.max(test_best_acc), ' - Best Test')
        print(epochs[np.argmax(test_best_acc)], ' - Best Number of Epochs' + '\n')
        print(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc)), ' - 95% CI Best Test')
        
        # Write best results to file
        f.write('Best Results\n')
        f.write(str(train_best_acc) + ' - Train' + '\n')
        f.write(str(np.max(train_best_acc)) + ' - Best Train' + '\n')
        f.write(str(test_best_acc) + ' - Test' + '\n')
        f.write(str(np.max(test_best_acc)) + ' - Best Test' + '\n')
        f.write(str(epochs[np.argmax(test_best_acc)]) + ' - Best Number of Epochs' + '\n')
        f.write(str(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc))) + ' - 95% CI Best Test' + 't\n')
        
        # Plot results
        plt.figure()
        exp1 = pd.DataFrame(epochs, columns=['No of Epochs'])
        exp1 = exp1.join(pd.DataFrame(test_best_acc, columns=['Accuracy']))
        sns.scatterplot(data=exp1, x='No of Epochs', y='Accuracy')
        plt.title('Epochs Experiment', size=24)
        
        # Return the best number of epochs
        return epochs[np.argmax(test_best_acc)]
        
def hidden_neurons_experiment(x_train, x_test, y_train, y_test, MaxRun, epoch=500):
    '''
    Get the optimal number of hidden neurons for the dataset

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    MaxRun : TYPE
        DESCRIPTION.
    epoch : TYPE, optional
        DESCRIPTION. The default is 500.

    Returns
    -------
    INT
        Optimal number of hidden neurons.

    '''
    # Define return values
    trainACC =  np.zeros(MaxRun)
    testACC =  np.zeros(MaxRun)
    hidden_neurons = [5,10,15,20,30,100,200,500]
    # hidden_neurons = [5,10]
    train_best_acc =  np.zeros(len(hidden_neurons))
    test_best_acc =  np.zeros(len(hidden_neurons))
    
    # Write output to file
    with open('experiment_hidden_neurons.log', 'w') as f:
        f.write('Log for Experiment Number 1 - Hidden Neurons' + '\n\n\n')
        # Fit model using number of experimental runs
        # i = 0
        for index, hid in enumerate(hidden_neurons):
        # while i < len(hidden_neurons):
            for run in range(0, MaxRun  ):
                
                print('experimental run number ' , run , ' with ' , hid , 'number of neurons') 
                # Define model
                model = Sequential()
                model.add(Dense(hid, input_dim=x_train.shape[1], activation='relu'))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))
                model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
                # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
                
                # Evaluate the model
                _, train_acc = model.evaluate(x_train, y_train, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
                trainACC[run] = train_acc
                testACC[run] = test_acc
            
            print(' print classification performance for each experimental' ) 
            print(trainACC, ' - Train')
            print(testACC, ' - Test')
            
            print(' print mean and std of training performance') 
            print(np.mean(trainACC), np.std(trainACC), ' - Train')
            # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            print(st.t.interval(0.95, len(trainACC)-1, loc=np.mean(trainACC), scale=st.sem(trainACC)), ' - 95% CI Train')
            print(np.mean(testACC), np.std(testACC),' - Test')
            print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Test')
            
            # Write to file
            f.write('experimental results with '+ str(hid) + ' number of neurons' + '\n\n')
            f.write(' print classification performance for each experimental' + '\n' ) 
            f.write(str(trainACC) + ' - Train' + '\n')
            f.write(str(testACC) + ' - Test' + '\n')
            
            f.write(' print mean and std of training performance' + '\n') 
            f.write(str(np.mean(trainACC)) + str(np.std(trainACC)) + ' - Train' + '\n')
            f.write(str(np.mean(testACC)) + str(np.std(testACC)) +' - Test' + '\n\n')
            
            # Save results 
            train_best_acc[index] = np.mean(trainACC)
            test_best_acc[index] = np.mean(testACC)
        # Print results to screen
        print('Best Results')
        print(train_best_acc, ' - Train')
        print(np.max(train_best_acc), ' - Best Train')
        print(test_best_acc, ' - Test')
        print(np.max(test_best_acc), ' - Best Test')
        print(hidden_neurons[np.argmax(test_best_acc)], ' - Best Hidden Nuerons' + '\n')
        print(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc)), ' - 95% CI Best Test')
        
        # Write best results to file
        f.write('Best Results\n')
        f.write(str(train_best_acc) + ' - Train' + '\n')
        f.write(str(np.max(train_best_acc)) + ' - Best Train' + '\n')
        f.write(str(test_best_acc) + ' - Test' + '\n')
        f.write(str(np.max(test_best_acc)) + ' - Best Test' + '\n')
        f.write(str(hidden_neurons[np.argmax(test_best_acc)]) + ' - Best Hidden Nuerons' + '\n')
        f.write(str(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc))) + ' - 95% CI Best Test' + 't\n')
        
        # Plot results
        plt.figure()
        exp1 = pd.DataFrame(hidden_neurons, columns=['Hidden Neurons'])
        exp1 = exp1.join(pd.DataFrame(test_best_acc, columns=['Accuracy']))
        sns.scatterplot(data=exp1, x='Hidden Neurons', y='Accuracy')
        plt.title('Hidden Neurons Experiment', size=24)
        
        # Return the best hidden  neurons
        return hidden_neurons[np.argmax(test_best_acc)], np.max(test_best_acc)

def learning_rate_experiment(x_train, x_test, y_train, y_test, MaxRun, epoch=500, hidden_neurons = 20):
    '''
     Get the optimal learning rate for SGD

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    MaxRun : TYPE
        DESCRIPTION.
    epoch : TYPE, optional
        DESCRIPTION. The default is 500.
    hidden_neurons : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    INT
        Optimal learning rate.

    '''
    # Define return values
    trainACC =  np.zeros(MaxRun)
    testACC =  np.zeros(MaxRun)
    # Optimal hidden neurons from experiment 1
    # hidden_neurons = 200
    learning_rate = [0.005,0.01,0.1,0.3,0.6,0.9]
    # hidden_neurons = [5,10]
    train_best_acc =  np.zeros(len(learning_rate))
    test_best_acc =  np.zeros(len(learning_rate))
    
    # Write output to file
    with open('experiment_learning_rate.log', 'w') as f:
        f.write('Log for Experiment Number 2 - Learning Rates' + '\n\n\n')
        # Fit model using number of experimental runs
        # i = 0
        for index, l_rate in enumerate(learning_rate):
        # while i < len(hidden_neurons):
            for run in range(0, MaxRun  ):
                
                print('experimental run number ' , run , ' with ' , l_rate , ' as the learning rate') 
                # Define model
                opt = SGD(learning_rate=l_rate)
                model = Sequential()
                model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))
                model.compile(opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
                # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
                
                # Evaluate the model
                _, train_acc = model.evaluate(x_train, y_train, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
                trainACC[run] = train_acc
                testACC[run] = test_acc
            
            print(' print classification performance for each experimental' ) 
            print(trainACC, ' - Train')
            print(testACC, ' - Test')
            
            print(' print mean and std of training performance') 
            print(np.mean(trainACC), np.std(trainACC), ' - Train')
            # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            print(st.t.interval(0.95, len(trainACC)-1, loc=np.mean(trainACC), scale=st.sem(trainACC)), ' - 95% CI Train')
            print(np.mean(testACC), np.std(testACC),' - Test')
            print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Test')
            
            # Write to file
            f.write('experimental results with '+ str(l_rate) + ' as the learning rate' + '\n\n')
            f.write(' print classification performance for each experimental run' + '\n' ) 
            f.write(str(trainACC) + ' - Train' + '\n')
            f.write(str(testACC) + ' - Test' + '\n')
            
            f.write(' print mean and std of training performance' + '\n') 
            f.write(str(np.mean(trainACC)) + str(np.std(trainACC)) + ' - Train' + '\n')
            f.write(str(np.mean(testACC)) + str(np.std(testACC)) +' - Test' + '\n\n')
            
            # Save results 
            train_best_acc[index] = np.mean(trainACC)
            test_best_acc[index] = np.mean(testACC)
        # Print results to screen
        print('Best Results')
        print(train_best_acc, ' - Train')
        print(np.max(train_best_acc), ' - Best Train')
        print(test_best_acc, ' - Test')
        print(np.max(test_best_acc), ' - Best Test')
        print(learning_rate[np.argmax(test_best_acc)], ' - Best Learning Rate' + '\n')
        print(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc)), ' - 95% CI Best Test')
        
        # Write best results to file
        f.write('Best Results\n')
        f.write(str(train_best_acc) + ' - Train' + '\n')
        f.write(str(np.max(train_best_acc)) + ' - Best Train' + '\n')
        f.write(str(test_best_acc) + ' - Test' + '\n')
        f.write(str(np.max(test_best_acc)) + ' - Best Test' + '\n')
        f.write(str(learning_rate[np.argmax(test_best_acc)]) + ' - Best Learning Rate' + '\n')
        f.write(str(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc))) + ' - 95% CI Best Test' + 't\n')
        
        # Plot results
        plt.figure()
        exp1 = pd.DataFrame(learning_rate, columns=['Learning Rate'])
        exp1 = exp1.join(pd.DataFrame(test_best_acc, columns=['Accuracy']))
        sns.scatterplot(data=exp1, x='Learning Rate', y='Accuracy')
        plt.title('Learning Rate Experiment', size=24)
        
        # Return optimal learning rate
        return learning_rate[np.argmax(test_best_acc)], np.max(test_best_acc)

def hidden_networks_experiment(x_train, x_test, y_train, y_test, MaxRun, epoch=500, hidden_neurons = 20):
    '''
    Get the optimal number of hidden layers for the dataset

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    MaxRun : TYPE
        DESCRIPTION.
    epoch : TYPE, optional
        DESCRIPTION. The default is 500.
    hidden_neurons : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    INT
        Optimal number of hidden layers.

    '''
    # Define return values
    trainACC =  np.zeros(MaxRun)
    testACC =  np.zeros(MaxRun)
    # Optimal hidden neurons
    # hidden_neurons = 200
    hidden_layers = [1,2,3]
    # hidden_neurons = [5,10]
    train_best_acc =  np.zeros(len(hidden_layers))
    test_best_acc =  np.zeros(len(hidden_layers))
    
    # Write output to file
    with open('experiment_hidden_networks.log', 'w') as f:
        f.write('Log for Experiment Number 3 - Different Hidden Layers' + '\n\n\n')
        # Fit model using number of experimental runs
        # i = 0
        for index, h_layer in enumerate(hidden_layers):
        # while i < len(hidden_neurons):
            for run in range(0, MaxRun  ):
                
                print('experimental run number ' , run , ' with ' , h_layer , ' hidden layer') 
                # Define model
                model = Sequential()
                model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
                if(h_layer == 2):
                    model.add(Dense(hidden_neurons, activation='relu'))
                if(h_layer == 3):
                    model.add(Dense(hidden_neurons, activation='relu'))
                    model.add(Dense(hidden_neurons, activation='relu'))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))
                # model.add(Dense(1, activation='relu'))
                model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'])
                # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
                
                # Evaluate the model
                _, train_acc = model.evaluate(x_train, y_train, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
                trainACC[run] = train_acc
                testACC[run] = test_acc
            
            print(' print classification performance for each experimental' ) 
            print(trainACC, ' - Train')
            print(testACC, ' - Test')
            
            print(' print mean and std of training performance') 
            print(np.mean(trainACC), np.std(trainACC), ' - Train')
            # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            print(st.t.interval(0.95, len(trainACC)-1, loc=np.mean(trainACC), scale=st.sem(trainACC)), ' - 95% CI Train')
            print(np.mean(testACC), np.std(testACC),' - Test')
            print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Test')
            
            # Write to file
            f.write('experimental results with '+ str(h_layer) + ' hidden layers' + '\n\n')
            f.write(' print classification performance for each experimental run' + '\n' ) 
            f.write(str(trainACC) + ' - Train' + '\n')
            f.write(str(testACC) + ' - Test' + '\n')
            
            f.write(' print mean and std of training performance' + '\n') 
            f.write(str(np.mean(trainACC)) + str(np.std(trainACC)) + ' - Train' + '\n')
            f.write(str(np.mean(testACC)) + str(np.std(testACC)) +' - Test' + '\n\n')
            
            # Save results 
            train_best_acc[index] = np.mean(trainACC)
            test_best_acc[index] = np.mean(testACC)
        # Print results to screen
        print('Best Results')
        print(train_best_acc, ' - Train')
        print(np.max(train_best_acc), ' - Best Train')
        print(test_best_acc, ' - Test')
        print(np.max(test_best_acc), ' - Best Test')
        print(hidden_layers[np.argmax(test_best_acc)], ' - Best Hidden Layers' + '\n')
        print(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc)), ' - 95% CI Best Test')
        
        # Write best results to file
        f.write('Best Results\n')
        f.write(str(train_best_acc) + ' - Train' + '\n')
        f.write(str(np.max(train_best_acc)) + ' - Best Train' + '\n')
        f.write(str(test_best_acc) + ' - Test' + '\n')
        f.write(str(np.max(test_best_acc)) + ' - Best Test' + '\n')
        f.write(str(hidden_layers[np.argmax(test_best_acc)]) + ' - Best Hidden Layers' + '\n')
        f.write(str(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc))) + ' - 95% CI Best Test' + 't\n')
        
        # Plot results
        plt.figure()
        exp1 = pd.DataFrame(hidden_layers, columns=['Hidden Layers'])
        exp1 = exp1.join(pd.DataFrame(test_best_acc, columns=['Accuracy']))
        sns.scatterplot(data=exp1, x='Hidden Layers', y='Accuracy')
        plt.title('Hidden Layers Experiment', size=24)
        
        # Return best hidden layers
        return hidden_layers[np.argmax(test_best_acc)], np.max(test_best_acc)

def optimiser_experiment(x_train, x_test, y_train, y_test, MaxRun, epoch=500, hidden_neurons = 20, h_layer = 2, learning_rate=0.1):
    '''
    Compare SGD and Adam, pass the optimal learning rate to be used by SGD

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    MaxRun : TYPE
        DESCRIPTION.
    epoch : TYPE, optional
        DESCRIPTION. The default is 500.
    hidden_neurons : TYPE, optional
        DESCRIPTION. The default is 20.
    h_layer : TYPE, optional
        DESCRIPTION. The default is 2.
    learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    None.

    '''
    # Using 1 hidden network as best result from experiment 3
    # Define return values
    trainACC =  np.zeros(MaxRun)
    testACC =  np.zeros(MaxRun)
    # Optimal hidden neurons from experiment 1
    # hidden_neurons = 200
    optimizers = ['sgd','adam']
    train_best_acc =  np.zeros(len(optimizers))
    test_best_acc =  np.zeros(len(optimizers))
    
    # Write output to file
    with open('experiment_optimisers.log', 'w') as f:
        f.write('Log for Experiment Number 4 - SGD and ADAM' + '\n\n\n')
        # Fit model using number of experimental runs
        # i = 0
        for index, opt in enumerate(optimizers):
        # while i < len(hidden_neurons):
            for run in range(0, MaxRun  ):
                
                print('experimental run number ' , run , ' with ' , opt) 
                # Define model
                model = Sequential()
                model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
                # model.add(Dense(hidden_neurons, activation='relu'))
                model.add(Dense(y_train.shape[1], activation='sigmoid'))
                if(h_layer == 2):
                    model.add(Dense(hidden_neurons, activation='relu'))
                if(opt == 'sgd'):
                    opt = SGD(learning_rate=learning_rate)
                model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer=opt)
                # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
                history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
                
                # Evaluate the model
                _, train_acc = model.evaluate(x_train, y_train, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                # print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
                trainACC[run] = train_acc
                testACC[run] = test_acc
            
            print(' print classification performance for each experimental' ) 
            print(trainACC, ' - Train')
            print(testACC, ' - Test')
            
            print(' print mean and std of training performance') 
            print(np.mean(trainACC), np.std(trainACC), ' - Train')
            # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            print(st.t.interval(0.95, len(trainACC)-1, loc=np.mean(trainACC), scale=st.sem(trainACC)), ' - 95% CI Train')
            print(np.mean(testACC), np.std(testACC),' - Test')
            print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Test')
            
            # Write to file
            f.write('experimental results with '+ str(opt) + '\n\n')
            f.write(' print classification performance for each experimental run' + '\n' ) 
            f.write(str(trainACC) + ' - Train' + '\n')
            f.write(str(testACC) + ' - Test' + '\n')
            
            f.write(' print mean and std of training performance' + '\n') 
            f.write(str(np.mean(trainACC)) + str(np.std(trainACC)) + ' - Train' + '\n')
            f.write(str(np.mean(testACC)) + str(np.std(testACC)) +' - Test' + '\n\n')
            
            # Save results 
            train_best_acc[index] = np.mean(trainACC)
            test_best_acc[index] = np.mean(testACC)
        # Print results to screen
        print('Best Results')
        print(train_best_acc, ' - Train')
        print(np.max(train_best_acc), ' - Best Train')
        print(test_best_acc, ' - Test')
        print(np.max(test_best_acc), ' - Best Test')
        print(optimizers[np.argmax(test_best_acc)], ' - Best Optimizer' + '\n')
        print(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc)), ' - 95% CI Best Test')
        
        # Write best results to file
        f.write('Best Results\n')
        f.write(str(train_best_acc) + ' - Train' + '\n')
        f.write(str(np.max(train_best_acc)) + ' - Best Train' + '\n')
        f.write(str(test_best_acc) + ' - Test' + '\n')
        f.write(str(np.max(test_best_acc)) + ' - Best Test' + '\n')
        f.write(str(optimizers[np.argmax(test_best_acc)]) + ' - Best Optimizer' + '\n')
        f.write(str(st.t.interval(0.95, len(train_best_acc)-1, loc=np.mean(train_best_acc), scale=st.sem(train_best_acc))) + ' - 95% CI Best Test' + 't\n')
        
        # Plot results
        plt.figure()
        exp1 = pd.DataFrame(optimizers, columns=['Optimisers'])
        exp1 = exp1.join(pd.DataFrame(test_best_acc, columns=['Accuracy']))
        sns.scatterplot(data=exp1, x='Optimisers', y='Accuracy')
        plt.title('Optimiser Experiment', size=24)

        return optimizers[np.argmax(test_best_acc)], np.max(test_best_acc)

def best_model(x_train, x_test, y_train, y_test, epoch=500, hidden_neurons = 100, h_layer = 1, opt = 'sgd'):
    '''
   Best model using adam optimiser

    Parameters
    ----------
    x_train : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    epoch : TYPE, optional
        DESCRIPTION. The default is 500.
    hidden_neurons : TYPE, optional
        DESCRIPTION. The default is 200.
    h_layer : TYPE, optional
        DESCRIPTION. The default is 1.
    opt : TYPE, optional
        DESCRIPTION. The default is adam.

    Returns
    -------
    None.

    '''
    #  Define output classes
    
   # Define model
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=x_train.shape[1], activation='relu'))
    # model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=AUC(curve="ROC"), optimizer=opt)
    # history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4000, verbose=0)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, verbose=0)
    
    # Evaluate the model
    _, train_auc = model.evaluate(x_train, y_train, verbose=0)
    _, test_auc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)

    test_scikit_auc = roc_auc_score(y_test, y_pred)
    
    print('Train: %.3f, Test keras: %.3f, Test scikit: %.3f' % (train_auc, test_auc, test_scikit_auc)) 
    
    # Confusion matrix
    # https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la
    
    y_score=np.argmax(y_pred, axis=1)
    y_true=np.argmax(y_test.to_numpy(), axis=1)
    cm = confusion_matrix(y_true, y_score)
    
    # create pandas dataframe
    class_names = ['Class 1',  'Class 2',  'Class 3',  'Class 4']
    dataframe_Confusion = pd.DataFrame(cm, index=class_names, columns=class_names)
    
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
    
    # ROC Curve
    y = label_binarize(y_test.to_numpy(), classes=class_names)
    n_classes = y.shape[1]
    plot_multiclass_roc(n_classes, y_test.to_numpy(), y_pred)
    
def main():
    # data_viz()
    # initial_load_cleanup()
    
    MaxRun = 1 # number of experimental runs
    models = ['Hidden Neurons', 'Learning Rate', 'Hidden Layer', 'Optimiser']
    testACC =  np.zeros(4)
    x_train, x_test, y_train, y_test = split_dataset()
    
    # Get the optimal number of epochs for the dataset
    # epoch = epochs_experiment(x_train, x_test, y_train, y_test, MaxRun)
    
    # # Get the optimal number of hidden neurons for the dataset
    hidden_neurons, testACC[0] = hidden_neurons_experiment(x_train, x_test, y_train, y_test, MaxRun)
    
    # # Get the optimal learning rate for SGD
    learning_rate, testACC[1] = learning_rate_experiment(x_train, x_test, y_train, y_test, MaxRun, hidden_neurons)
    
    # # Get the optimal number of hidden layers for the dataset
    h_layer, testACC[2] = hidden_networks_experiment(x_train, x_test, y_train, y_test, MaxRun, hidden_neurons)
    # h_layer, testACC[2] = hidden_networks_experiment(x_train, x_test, y_train, y_test, MaxRun)
    
    # Compare SGD and Adam, pass the optimal learning rate to be used by SGD
    opt, testACC[3] = optimiser_experiment(x_train, x_test, y_train, y_test, MaxRun, hidden_neurons, h_layer, learning_rate)
    # opt, testACC[3] = optimiser_experiment(x_train, x_test, y_train, y_test, MaxRun, hidden_neurons = 200, h_layer = 1)

    # Print best model output from experiments
    
    print(np.max(testACC), ' - Best Test')
    print(models[np.argmax(testACC)], ' - Best Model' + '\n')
    print(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC)), ' - 95% CI Best Test')
    with open('best_model.log', 'w') as f:
        f.write(str(testACC) + ' - Test' + '\n')
        f.write(str(np.max(testACC)) + ' - Best Test' + '\n')
        f.write(str(models[np.argmax(testACC)]) + ' - Best Model' + '\n')
        f.write(str(st.t.interval(0.95, len(testACC)-1, loc=np.mean(testACC), scale=st.sem(testACC))) + ' - 95% CI Best Test' + 't\n')
    
    best_model(x_train, x_test, y_train, y_test)
    
if __name__ == '__main__':
    main()