"""
Author : Pedram Mirshahreza
CS 596 Term Project
Created: Dec 12 2019
"""

# Artificial Neural Network Model

# Importing the libraries
import itertools 
import keras
import numpy             as np
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from   keras.models import Sequential
from   keras.layers import Dense

from   sklearn.model_selection import train_test_split
from   sklearn.utils           import class_weight
from   sklearn.metrics         import confusion_matrix 
from   sklearn.metrics         import accuracy_score 
from   sklearn.metrics         import f1_score
from   sklearn.metrics         import recall_score 
from   sklearn.metrics         import precision_score



# ----------- Part 0 - Setting the Hyper Parameters --#
#The hyper parameters set for the ANN
test_set_ratio = 0.2
Epochs         = 10
Batch_size     = 10


# ----------- Part 1 - Handling the dataset -----------#

# Importing the dataset
file_name = 'convertcsv.csv'
dataset = pd.read_csv(file_name)

# remove unnecessary columns
del dataset["pr_number"]  # this is just ID .. not needed

# Assign X and y
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_set_ratio, 
                                                    random_state = 0)


# ----------- Part 2 - ANN model -----------#

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, 
                     kernel_initializer = 'uniform', 
                     activation         = 'relu'   , 
                     input_dim          = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(6, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu'))



# Adding the output layer
classifier.add(Dense(units = 1, 
                     kernel_initializer = 'uniform', 
                     activation         = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', 
                        loss = 'binary_crossentropy', 
                     metrics = ['accuracy'])

# Creating the class weights to try to balance data
Class_weights = class_weight.compute_class_weight('balanced', 
                                                  np.unique(y_train), 
                                                            y_train  )


# Fitting the ANN to the Training set

classifier.fit(X_train, y_train         , 
               batch_size   = Batch_size, 
               epochs       = Epochs    , 
               class_weight = Class_weights)

#----------- Part 3 - Making predictions and evaluating the model -----------#

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Reporting Metrics:
# Confusion Matrix :
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n-----------------")
print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in cm]))

print("\n====================\n")
print("Metrics:\n--------")

recal_class_0     = cm[0][0]/float(cm[0][0] + cm[0][1])
recal_class_1     = cm[1][1]/float(cm[1][0] + cm[1][1])
Precision_class_1 = cm[1][1]/float(cm[0][1] + cm[1][1])
accuracy          =  accuracy_score(y_test, y_pred)
recall_score      =    recall_score(y_test, y_pred)
precision_score   = precision_score(y_test, y_pred)
f1_score          =        f1_score(y_test, y_pred)

print("The model's accuracy is  {}%".format(round(accuracy       *100,3)))
print("The model's recall is    {}%".format(round(recall_score   *100,3)))
print("The model's recall for class 0 is    {}%".format(round(
                                                 recal_class_0   *100,3)))
print("The model's recall for class 1 is    {}%".format(round(
                                                 recal_class_1   *100,3)))
print("The model's Precision for class 1 is  {}%".format(round(
                                                Precision_class_1*100,3)))
print("The model's precision is {}%".format(round(precision_score*100,3)))
print("The model's f1-score is  {}%".format(round(f1_score       *100,3)))

def plot_ConfMat(methodname,cm, classes, normalize = False,
                 title='Confusion matrix',
                 cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization for ', 
              methodname,' method')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

plot_ConfMat("3 layer ANN",cm, ['Class 0', 'Class 1'], normalize = False,
                 title='Confusion matrix',
                 cmap=plt.cm.Blues)
