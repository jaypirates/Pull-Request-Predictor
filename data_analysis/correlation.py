# Author: Kunal Nain

import numpy as np
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# Opening the norm_output.json file which contains the normalized data
os.chdir('..')
filepath = os.getcwd()
filepath = filepath + "/data/norm_output.json"
f = open(filepath, "r")

# Loading the files data as JSON
json_data = json.loads(f.read())

# Creating lists to hold the data
x_status_check = []
x_user_type = []
x_user_contr = []
x_comment_count = []
x_time_duration = []
y = []

# Parsing the data and loading the lists
for data_object in json_data:
    x_status_check.append(data_object['pr_status_check'])
    x_user_type.append(data_object['pr_user_type'])
    x_user_contr.append(data_object['pr_user_contr'])
    x_comment_count.append(data_object['pr_comment_count'])
    x_time_duration.append(data_object['pr_time_duration'])
    y.append(data_object['pr_is_merged'])

# Converting the lists into Numpy Array
x_array_status_check = np.array(x_status_check)
x_array_user_type = np.array(x_user_type)
x_array_user_contr = np.array(x_user_contr)
x_array_comment_count = np.array(x_comment_count)
x_array_time_duration = np.array(x_time_duration)

y_array = np.array(y)

# Reshaping the array so that it is compatible with the Linear Regression method
x_array_status_check = x_array_status_check.reshape(-1, 1)
x_array_user_type = x_array_user_type.reshape(-1, 1)
x_array_user_contr = x_array_user_contr.reshape(-1, 1)
x_array_comment_count = x_array_comment_count.reshape(-1, 1)
x_array_time_duration = x_array_time_duration.reshape(-1, 1)

y_array = y_array.reshape(-1, 1)

# Calculating the R-Squared
reg_status_check = LinearRegression().fit(x_array_status_check, y_array)
reg_user_type = LinearRegression().fit(x_array_user_type, y_array)
reg_user_contr = LinearRegression().fit(x_array_user_contr, y_array)
reg_comment_count = LinearRegression().fit(x_array_comment_count, y_array)
reg_time_duration = LinearRegression().fit(x_array_time_duration, y_array)

print("PR Status Check R-Squared Score = ", reg_status_check.score(x_array_status_check, y_array))
print("PR User Type R-Squared Score = ", reg_user_type.score(x_array_user_type, y_array))
print("PR User Contribution R-Squared Score = ", reg_user_contr.score(x_array_user_contr, y_array))
print("PR Comment Count R-Squared Score = ", reg_comment_count.score(x_array_comment_count, y_array))
print("PR Time Duration R-Squared Score = ", reg_time_duration.score(x_array_time_duration, y_array))

# Plotting the data
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(x_array_status_check, y_array)
axs[0, 0].set_title('Status Check R-Squared')
axs[0, 1].plot(x_array_user_type, y_array, 'tab:orange')
axs[0, 1].set_title('User Type R-Squared')
axs[1, 0].plot(x_array_user_contr, y_array, 'tab:green')
axs[1, 0].set_title('User Contribution R-Squared')
axs[1, 1].plot(x_array_comment_count, y_array, 'tab:red')
axs[1, 1].set_title('Comment Count R-Squared')
axs[2, 0].plot(x_array_time_duration, y_array, 'tab:purple')
axs[2, 0].set_title('Time Duration R-Squared')
plt.show()
