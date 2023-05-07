from tkinter import *
import tkinter 
''' #Tkinter is the de facto way in Python to create 
#Graphical User interfaces (GUIs) and is included in all 
standard Python Distributions. '''
from tkinter import filedialog
import numpy as np #NumPy 
''' offers comprehensive mathematical functions, random 
number generators, linear algebra routines, Fourier 
transforms, and more.'''
from tkinter.filedialog import askopenfilename
import pandas as pd
'''#pandas is a fast,powerful,flexible and easy to use open 
source data analysis and manipulation tool,built on top 
of the Python programming language.'''
from tkinter import simpledialog
import matplotlib.pyplot as plt 
'''#Matplotlib is a comprehensive library for creating 
static, animated, and interactive visualizations in 
Python.'''
import os #OS 
'''module provides the facility to establish the interaction 
between the user and the operating system.
from keras.utils.np_utils import to_categorical #Keras 
is a powerful and easy-to-use free open source Python 
library for developing and evaluating deep learning 
models.'''
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import to_categorical
import pickle 
#Python pickle module is used for serializing and deserializing a Python object structure.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.svm import SVC
main = tkinter.Tk()
main.title("Predicting Memory Compiler Performance Outputs using Feed-Forward Neural Networks")
main.geometry("1000x650")
global filename
global classifier 
#Variables that are created outside of a function (as in all of the examples above) are known as global variables.
#global X, Y, X_train, X_test, y_train, y_test 
#Global variables can be used both inside of functions and outside.
global dataset
global error
def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n') 
# Code for uploading the dataset file.
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True,xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
def preprocess():
    text.delete('1.0', END)
    global dataset
    global X, Y, X_train, X_test, y_train, y_test
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y = Y.astype("int") 
# For pre-processing the dataset into 'training' and 'testing' sets
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"Total records used to train Neural Networks: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test Neural Networks : "+str(X_test.shape[0])+"\n")
 
#--------------------------FEED FORWARD NEURAL NETWORK ---------------------------------
def runFeedForward():
    text.delete('1.0', END)
    global X, Y, error
    error = []
    Y1 = to_categorical(Y)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)

    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()
    else:
        classifier = Sequential()
        classifier.add(Dense(200, activation='relu', 
        input_shape=(X_train1.shape[1],)))
        classifier.add(Dense(150, activation='relu'))
        classifier.add(Dense(100, activation='relu'))
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax')) #'softmax' [output layer] - it is used for multi-class classification
        classifier.compile(optimizer = 'adam', loss = 
        'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train1, y_train1, batch_size=8, epochs=1000, validation_data=(X_test1, y_test1), shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5') 
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:json_file.write(model_json)
        json_file.close = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        err = 100 - (acc[999] * 100)
        error.append(err)
        print(classifier.summary())
        text.insert(END,"Feed Forward Neural Network Error Rate: "+str(err)+"\n\n")
        predict = classifier.predict(X_test1)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test1, axis=1)
        LABELS = ['Area','Memory Leakage']
        conf_matrix = confusion_matrix(y_test1, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("Predicting Memory Feed Forward Network Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show() 
#-------------------------- GRADIENT BOOSTING ---------------------------------
def runGradientBoosting():
    global X_train, X_test, y_train, y_test
    gb = GradientBoostingClassifier()
    gb.fit(X_train,y_train)
    predict = gb.predict(X_test)

    accuracy = accuracy_score(predict, y_test) * 100
    err = 100 - accuracy
    error.append(err)
    text.insert(END,"Gradient Boosting Predicting Memory Error Rate: "+str(err)+"\n\n")
    LABELS = ['Area','Memory Leakage']
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, 
    yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Predicting Memory Gradient Boosting Algorithm Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 
#-------------------------- RANDOM FOREST ---------------------------------
def runRandomForest():
    global X_train, X_test, y_train, y_test
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    predict = rf.predict(X_test)
    accuracy = accuracy_score(predict, y_test) * 100
    err = 100 - accuracy
    error.append(err)
    text.insert(END,"Random Forest Predicting Memory Error Rate: "+str(err)+"\n\n")
    LABELS = ['Area','Memory Leakage']
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Predicting Memory using Random Forest Algorithm Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
#-------------------------- LOGISTIC REGRESSION ---------------------------------
def runLogisticRegression():
    global X_train, X_test, y_train, y_test
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    predict = lr.predict(X_test)
    accuracy = accuracy_score(predict, y_test) * 100
    err = 100 - accuracy
    error.append(err)
    text.insert(END,"Logistic Regression Predicting Memory Error Rate: "+str(err)+"\n\n")
    LABELS = ['Area','Memory Leakage']
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Predicting Memory using Logistic Regression Algorithm Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
#-------------------------- POLYNOMIAL REGRESSION ---------------------------------
def runPolynomialRegression():
    global X_train, X_test, y_train, y_test
    pr = SVC()
    pr.fit(X_train,y_train)
    predict = pr.predict(X_test)
    accuracy = accuracy_score(predict, y_test) * 100
    err = 100 - accuracy
    error.append(err)
    text.insert(END,"Polynomial Regression Predicting Memory Error Rate: "+str(err)+"\n\n")
    LABELS = ['Area','Memory Leakage']
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Predicting Memory using Polynomial Regression Algorithm Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
#-------------------------- K Means NETWORK ---------------------------------
def runKMN():
    global X_train, X_test, y_train, y_test
    kmn = KNeighborsClassifier()
    kmn.fit(X_train,y_train)
    predict = kmn.predict(X_test)
    accuracy = accuracy_score(predict, y_test) * 100
    err = 100 - KMeans.reg
    error.append(err)
    text.insert(END,"KMN Predicting Memory Error Rate: "+str(err)+"\n\n")
    LABELS = ['Area','Memory Leakage']
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("Predicting Memory using KMN Algorithm Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
#-------------------------- Code for Comparison Graph ---------------------------------
def graph():
    height = error
    bars = ('Feed-Forward','Gradient Boosting','Random Forest','Logistic Regression','Polynomial Regression','Convolution 2d')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Predicting Memory Performance Error Rate")
    plt.show()
def close():
    main.destroy()
#-------------------------- GUI --------------------------------- 
font = ('Bahnschrift', 18, 'bold')
title = Label(main, text='Predicting Memory Compiler Performance Outputs using Deep Learning Neural Networks', 
justify=LEFT)
title.config(bg='black', fg='white')
title.config(font=font) 
title.config(height=3, width=120)
title.place(x=100,y=5)
title.pack()
font1 = ('Bahnschrift', 12, 'bold')
uploadButton = Button(main, text="Upload PPA Memory Optimization Dataset", command=upload)
uploadButton.place(x=30,y=100)
uploadButton.config(font=font1) 
gbButton = Button(main, text="Train Gradient Boosting Algorithm", command=runGradientBoosting)
gbButton.place(x=30,y=150)
gbButton.config(font=font1)
rfButton = Button(main, text="Train Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=320,y=150)
rfButton.config(font=font1)
lrButton = Button(main, text="Train Logistic Regression Algorithm", command=runLogisticRegression)
lrButton.place(x=590,y=150)
lrButton.config(font=font1)
kmnButton = Button(main, text="Train KMN Algorithm", command=runKMN)
kmnButton.place(x=370,y=200)
kmnButton.config(font=font1)
prButton = Button(main, text="Train Polynomial Regression Algorithm", command=runPolynomialRegression)
prButton.place(x=30,y=200)
prButton.config(font=font1)
closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=890,y=200)
closeButton.config(font=font1)
font1 = ('Bahnschrift', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 
main.config(bg='#1F618D')
main.mainloop()