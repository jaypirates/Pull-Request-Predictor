# Author: Kunal Nain

import numpy as np
import json
import matplotlib.pyplot as plt
import os
from collections import Counter

# Opening the output.json file which contains the data
os.chdir('..')
filepath = os.getcwd()
filepath = filepath + "/data/output.json"
f = open(filepath, "r")

# Loading the file data into a JSON object
json_data = json.loads(f.read())

# Creating lists to hold the training and testing data
x_status_check = []
x_user_type = []
x_user_contr = []
x_comment_count = []
x_time_duration = []
y = []

# Populating the lists
for data_object in json_data:
    x_status_check.append(data_object['pr_status_check'])
    x_user_type.append(data_object['pr_user_type'])
    x_user_contr.append(data_object['pr_user_contr'])
    x_comment_count.append(data_object['pr_comment_count'])
    x_time_duration.append(data_object['pr_time_duration'])
    y.append(data_object['pr_is_merged'])

# Converting the python lists into Numpy arrays
x_array_status_check = np.array(x_status_check)
x_array_user_type = np.array(x_user_type)
x_array_user_contr = np.array(x_user_contr)
x_array_comment_count = np.array(x_comment_count)
x_array_time_duration = np.array(x_time_duration)

# Counting the frequency of the keys
cnt_status_check = Counter(x_array_status_check)
cnt_user_type = Counter(x_array_user_type)
cnt_user_contr = Counter(x_array_user_contr)
cnt_comment_count = Counter(x_array_comment_count)
cnt_time_duration = Counter(x_array_time_duration)

print(cnt_status_check)
print(cnt_user_type)
print(cnt_user_contr)
print(cnt_comment_count)
print(cnt_time_duration)

# Plotting the data
fig, axs = plt.subplots(5, 1)
axs[0].bar(cnt_status_check.keys(), cnt_status_check.values(), width=1)
axs[0].set_title('Status Check Frequency')
axs[1].bar(cnt_user_type.keys(), cnt_user_type.values(), color='orange', width=1)
axs[1].set_title('User Type Frequency')
axs[2].bar(cnt_user_contr.keys(), cnt_user_contr.values(), color='green', width=1)
axs[2].set_title('User Contribution Frequency')
axs[3].bar(cnt_comment_count.keys(), cnt_comment_count.values(), color='red', width=1)
axs[3].set_title('Comment Count Frequency')
axs[4].bar(cnt_time_duration.keys(), cnt_time_duration.values(), color='purple', width=1000)
axs[4].set_title('Time Duration Frequency')
plt.tight_layout()
plt.savefig('frequency.png', dpi=300)
plt.show()
plt.close()
