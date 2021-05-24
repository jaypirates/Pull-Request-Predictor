# Name: Derek Opel

import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from conf_matrix import func_confusion_matrix

## Initialize matrix to store the normalized data in
data = np.zeros((100,7))
row = 0

## Open file and store the data in a 100x7 matrix
with open('norm_output.json', 'r') as json_file:
        data_mtx = json.load(json_file)

        for obj in data_mtx:
                data[row] = list(obj.values())
                row += 1
n = 100

X = data[:,1:6]
y = data[:,6]

###################################################################################################################################
#### Model selection using cross-validation to find the optimal hyperparameters C and kernel types(linear,RBF,polynomial) (MODEL 1) #### 
###################################################################################################################################

# Plot the average error--using 5 fold cross-validation--for different values of C under each kernel  

fig = plt.figure()
rows = 1
cols = 3

c_range = [0.1,0.5,2,4,6,8,10]

kernel_types = ['linear', 'poly', 'rbf']

cv_scores = [[],[],[]]

for index, kernel_value in enumerate(kernel_types):
	
	for c_value in c_range:
		model = svm.SVC(kernel=kernel_value, C=c_value, gamma='auto')
		scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')	
		cv_scores[index].append(scores.mean())

	avg_error = [1-x for x in cv_scores[index]]

	fig.add_subplot(rows,cols,index + 1)
	plt.plot(c_range, avg_error)
	plt.title(kernel_value.upper())
	plt.xlabel('C values')
	plt.ylabel('Avg. Misclassification Error')

plt.tight_layout()
plt.show()

## Select the best model and apply it over the testing dataset

# Both the linear and RBF kernel were tied with the lowest errors
best_kernel = 'linear'

# The linear kernel had many Cs that were "best"
best_c = 2

## Split data into training/testing sets

split = np.random.permutation(n)

x_train = data[split[:50],1:6]
y_train = data[split[:50],6].ravel()

x_test = data[split[50:100],1:6]
y_test = data[split[50:100],6].ravel()

svm_model1 = svm.SVC(kernel=best_kernel, C=best_c)
svm_model1.fit(X=x_train, y=y_train)

## Evaluate the results with the confusion matrix, accuracy, recall per class, and precision per class
y_pred = svm_model1.predict(x_test)

conf_mtx, acc, recall_arr, pr_arr = func_confusion_matrix(y_test,y_pred)

print("Confusion Matrix with optimal parameters:")
print(conf_mtx)
print("Accuracy:\n{}".format(acc))
print("Per-Class Precision\n{}".format(pr_arr))
print("Per-Class Recall:\n{}".format(recall_arr))
print()

####################################################
#### Implement a Bagging classifier (MODEL 2) ####
####################################################

SVM = svm.LinearSVC(random_state=42)
svm_model2 = BaggingClassifier(base_estimator=SVM, n_estimators=45,random_state=214)
svm_model2.fit(X, y)

## Evaluate the results with the confusion matrix, accuracy, recall per class, and precision per class
y_pred = svm_model2.predict(x_test)

conf_mtx, acc, recall_arr, pr_arr = func_confusion_matrix(y_test,y_pred)

print("Confusion Matrix with Bagging:")
print(conf_mtx)
print("Accuracy:\n{}".format(acc))
print("Per-Class Precision\n{}".format(pr_arr))
print("Per-Class Recall:\n{}".format(recall_arr))
print()

## Compare the average accuracy between the svm model with the optimal hyperparameters and the bagging classifier, using 5-fold cross-validation
model1_scores = cross_val_score(svm_model1, X, y, cv=5, scoring='accuracy')
model2_scores = cross_val_score(svm_model2, X, y, cv=5, scoring='accuracy')

print("SVM (using optimal hyperparameters) Average Accuracy: {}\n".format(model1_scores.mean()))
print("Bagging Classifer Average Accuracy: {}\n".format(model2_scores.mean()))

