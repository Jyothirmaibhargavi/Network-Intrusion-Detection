from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.models import Sequential
from keras.layers import Dense, Dropout



main = tkinter.Tk()
main.title("Network Intrusion Detection")
main.geometry("1300x1200")

global filename
global labels 
global columns
global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, ann_acc, classifier

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def splitdataset(balance_data): 
    X = balance_data.values[:, 0:38] 
    Y = balance_data.values[:, 38]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "NSL-KDD-Dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def preprocess(): 
    global labels, columns, filename
    
    text.delete('1.0', END)
    
    balance_data = pd.read_csv(filename)
    dataset = ''
    index = 0
    cols = ''

    for index, row in balance_data.iterrows():
        for i in range(len(row)):  # Iterate only up to available columns
            if isfloat(row.iloc[i]):
                dataset += str(row.iloc[i]) + ','
                if index == 0:
                    cols += f"Column{i},"  # Dynamically generate column names
        
        # Handle label column dynamically
        label_col = row.iloc[-1]  # Last column as label
        dataset += '0' if label_col == 'normal' else '1'

        if index == 0:
            cols += "Label"
        dataset += '\n'
        index = 1  # Ensure only first row gets column names

    # Save processed data
    with open("clean.txt", "w") as f:
        f.write(cols + "\n" + dataset)

    text.insert(END, "Preprocessing complete. Data saved to clean.txt\n")


def generateModel():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    global balance_data
    balance_data = pd.read_csv("clean.txt") 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(balance_data)
    text.insert(END,"Train & Test Model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(balance_data))+"\n")
    text.insert(END,"Split Training Size : "+str(len(X_train))+"\n")
    text.insert(END,"Split Test Size : "+str(len(X_test))+"\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    return accuracy  

def runSVM():
    text.delete('1.0', END)
    global svm_acc
    global classifier
    global X, Y, X_train, X_test, y_train, y_test
    total = X_train.shape[1];
    X_train1 = SelectKBest(score_func=chi2, k=15).fit_transform(X_train, y_train)
    X_test1 = SelectKBest(score_func=chi2,k=15).fit_transform(X_test,y_test)
    text.insert(END,"Total Features : "+str(total)+"\n")
    text.insert(END,"Features set reduce after applying features selection concept : "+str((total - X_train.shape[1]))+"\n\n")
    cls = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix')
    classifier = cls

def runRF():
    text.delete('1.0', END)
    global rf_acc
    global X, Y, X_train, X_test, y_train, y_test
    
    total = X_train.shape[1]
    X_train1 = SelectKBest(score_func=chi2, k=15).fit_transform(X_train, y_train)
    X_test1 = SelectKBest(score_func=chi2, k=15).fit_transform(X_test, y_test)
    
    text.insert(END, "Total Features : " + str(total) + "\n")
    text.insert(END, "Features set reduced after applying feature selection: " + str(total - X_train1.shape[1]) + "\n\n")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train1, y_train)
    
    prediction_data = prediction(X_test1, rf)
    rf_acc = cal_accuracy(y_test, prediction_data, 'Random Forest Accuracy, Classification Report & Confusion Matrix')



def runDT():
    text.delete('1.0', END)
    global dt_acc
    global X, Y, X_train, X_test, y_train, y_test

    total = X_train.shape[1]
    
    # Apply feature selection
    X_train1 = SelectKBest(score_func=chi2, k=15).fit_transform(X_train, y_train)
    X_test1 = SelectKBest(score_func=chi2, k=15).fit_transform(X_test, y_test)

    text.insert(END, "Total Features : " + str(total) + "\n")
    text.insert(END, "Features set reduced after applying feature selection: " + str(total - X_train1.shape[1]) + "\n\n")

    # Train Decision Tree with reduced features
    cls = DecisionTreeClassifier(random_state=0)
    cls.fit(X_train1, y_train)
    
    prediction_data = prediction(X_test1, cls)
    dt_acc = cal_accuracy(y_test, prediction_data, 'Decision Tree Accuracy, Classification Report & Confusion Matrix')



def runANN():
    text.delete('1.0', END)
    global ann_acc
    global X, Y, X_train, X_test, y_train, y_test 
    total = X_train.shape[1];
    X_train = SelectKBest(score_func=chi2,k=25).fit_transform(X_train, y_train)
    X_test = SelectKBest(score_func=chi2,k=25).fit_transform(X_test,y_test)
    text.insert(END,"Total Features : "+str(total)+"\n")
    text.insert(END,"Features set reduce after applying features selection concept : "+str((total - X_train.shape[1]))+"\n\n")
    model = Sequential()
    model.add(Dense(64, input_dim=25, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200, batch_size=32)
    _, ann_acc = model.evaluate(X_train, y_train)
    ann_acc = ann_acc*100
    text.insert(END,"ANN Accuracy : "+str(ann_acc)+"\n\n")
    
from tkinter import messagebox
import matplotlib.pyplot as plt

from tkinter import messagebox
import matplotlib.pyplot as plt

def detectAttack():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    
    # Upload the test data file
    filename = filedialog.askopenfilename(initialdir="NSL-KDD-Dataset")
    test = pd.read_csv(filename)
    text.insert(END, filename + " test file loaded\n")
    
    # Predict the test data using the trained classifier
    y_pred = classifier.predict(test)

    attack_count = 0
    normal_count = 0
    
    # Flag to track if alert has been shown
    alert_shown = False
    
    # Iterate over test data and display results in the text box
    with open("detection_results.txt", "w") as f:
        for i in range(len(test)):
            if str(y_pred[i]) == '1.0':
                result = "X=%s, Predicted=%s" % (X_test[i], 'Infected. Detected Anomaly Signatures')
                attack_count += 1
                if not alert_shown:
                    # Show alert only once, even if there are many attacks
                    messagebox.showwarning("Intrusion Detected!", "An attack has been detected in the uploaded test data!")
                    alert_shown = True
            else:
                result = "X=%s, Predicted=%s" % (X_test[i], 'Normal Signatures')
                normal_count += 1
            text.insert(END, result + "\n\n")
            f.write(result + "\n")
    
        # Pie chart summary
    labels = ['Normal', 'Attack']
    sizes = [normal_count, attack_count]
    colors = ['green', 'red']
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Intrusion Detection Summary')
    plt.axis('equal')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def graph():
    height = [svm_acc, ann_acc, rf_acc, dt_acc]  # Accuracy values
    bars = ['SVM', 'ANN', 'RF', 'DT']
    y_pos = np.arange(len(bars))

    plt.figure(figsize=(8, 6))  # Set proper figure size
    bars_plot = plt.bar(y_pos, height, color='orange', edgecolor='black', width=0.6)

    # Add accuracy values on top of each bar
    for bar in bars_plot:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.xticks(y_pos, bars, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title('Comparison of Algorithms', fontsize=15)
    plt.ylim(0, 105)  # Adjust to give space above bars

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Network Intrusion Detection using Supervised Machine Learning Technique with Feature Selection')
title.config(bg='PaleGreen2', fg='Khaki4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload NSL KDD Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

preprocess = Button(main, text="Preprocess Dataset", command=preprocess)
preprocess.place(x=700,y=200)
preprocess.config(font=font1) 

model = Button(main, text="Generate Training Model", command=generateModel)
model.place(x=700,y=250)
model.config(font=font1) 

runsvm = Button(main, text="Run SVM Algorithm", command=runSVM)
runsvm.place(x=700,y=300)
runsvm.config(font=font1) 

rfButton = Button(main, text="Run Random Forest", command=runRF)
rfButton.place(x=700, y=400)
rfButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree", command=runDT)
dtButton.place (x=700, y=450)
dtButton.config(font=font1)


annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=700,y=350)
annButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=700,y=500)
graphButton.config(font=font1) 

attackButton = Button(main, text="Upload Test Data & Detect Attack", command=detectAttack)
attackButton.place(x=700,y=550)
attackButton.config(font=font1) 



font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='PeachPuff2')
main.mainloop()
