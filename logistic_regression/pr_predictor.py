# AUTHOR : JAY PRABHUBHAI PATEl

"""PR_predictor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fhgiTwXRJR22jB7HKXLHWxJeCBldD4XF

# **Importing Essentials libraries**
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("convertcsv.csv")
df = df.drop("pr_number",axis=1)
print(df.head())
print()

"""# **Let's Plot Correlation Graph**"""

corr_matrix = df.corr()
print(corr_matrix["pr_is_merged"].sort_values(ascending = False))

plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix,vmin=-1,vmax=1,square=True, linewidths=.5,fmt='.2f',cmap="BrBG")

"""# **Training and Testing a Model**"""

X=df.drop("pr_is_merged",axis=1).values
Y = df[["pr_is_merged"]].values

X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size = 0.2,random_state = 42)

lr = LogisticRegression(random_state=42)
lr.fit(X_train,Y_train)
print()
print("==================== : Before HyperParameter Tuning : ===============")
print("Training Accuracy: ", lr.score(X_train,Y_train))
print("Testing Accuracy: ", lr.score(X_test,Y_test))
print()

"""# **Hyper-Parameter Tuning**"""

penalty = ['l1', 'l2']
solver=['liblinear', 'saga']
C = np.logspace(-3, 3, 7)
hyperparameters = dict(C=C, penalty=penalty,solver=solver)

clf = LogisticRegression(random_state=42)

clf = GridSearchCV(clf, hyperparameters, cv=5)

best_model = clf.fit(X_train, Y_train)

print("==================== : After HyperParameter Tuning : ===============")
print("Training Accuracy: ", clf.score(X_train,Y_train))
print("Testing Accuracy: ", clf.score(X_test,Y_test))
print()
Y_pred = best_model.best_estimator_.predict(X_test)

"""# **Evaluation of a Model**"""


cm = confusion_matrix(Y_test,Y_pred)
print("==================== : Evaluations : ===============")
print("Confusion Matrix:")
print(pd.DataFrame(cm))
print(classification_report(Y_test,Y_pred))
