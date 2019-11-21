# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:24 2019
@author: Aditya
"""

import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_excel (r"C:\Users\Aditya\Documents\ig_views.xlsx")

#Read the required data from dataset
df_needed = df[['Day','23-24hr']];
df_needed = df_needed.dropna(how = 'all');

#Extract the column to be modified
df_gt113 = df_needed.iloc[:,1]

#Apply lambda function to set 0/1
df_gt113 = df_gt113.apply(lambda x: 1 if x>113 else 0)
df_needed['23-24hr'] = df_gt113
df_needed.columns = ['Day','Original_views']

#PLOT the frequency of views>113
ax = df_needed.plot(kind = 'bar', color = 'blue', label = 'Greater than 113 views', figsize = (15,5))
ax.set_xticklabels(df_needed['Day']) ;

#SAVE THE PLOT FIGURE IN CURRENT DIRECTORY
fig = ax.get_figure();
fig.savefig('Original_plot')

#Number of 1's (minority class)
n1 = df_gt113.value_counts()[1];
print('Current status of minority class (views>113): ', n1*100/len(df_gt113), '%');
#One hot encode weekdays

# One-hot encode weekday names into integer values
label_encoder = LabelEncoder();
integer_encoded = label_encoder.fit_transform(df_needed['Day']);

#Apply SMOTE algorithm
smote_handle = SMOTE();
#Give the two classes, less than and greater than 113
x = integer_encoded
y = df['23-24hr'].dropna()

#Using lambda function to quickly switch all values less than 113 to 0
time_freq = y.apply(lambda x: 1 if x>113 else 0);
day_resampled, time = smote_handle.fit_resample(x.reshape(-1,1),time_freq);

#Return One-hot encoded weekday names into strings
inverted_days = label_encoder.inverse_transform(day_resampled.ravel());

#Create a dataframe out of ndarrays
data = {'Day':inverted_days,'SMOTE_Oversampled_views': time};
df_smote = pd.DataFrame(data);

#PLOT the frequency of views>113 after SMOTE
ax = df_smote.plot(kind = 'bar', color = 'blue', label = 'Greater than 113 views', figsize = (15,5))
ax.set_xticklabels(df_smote['Day']) 

#SAVE THE PLOT FIGURE IN CURRENT DIRECTORY
fig = ax.get_figure()
fig.savefig('SMOTE_Oversampled_plot')

#New number of 1's (minority class)
n1 = df_smote['SMOTE_Oversampled_views'].value_counts()[1]
print('New status of minority class (views>113): ', n1*100/len(df_smote), '%')

#Apply ROSE algorithm

rose_handle = RandomOverSampler(return_indices=True)
X_rose, y_rose, id_rose = rose_handle.fit_sample(x.reshape(-1,1),time_freq)

#Return One-hot encoded weekday names into strings
inverted_days = label_encoder.inverse_transform(X_rose.ravel());

#Create a dataframe out of ndarrays
data = {'Day':inverted_days,'ROSE_Oversampled_views': y_rose};
df_rose = pd.DataFrame(data);

#PLOT the frequency of views>113 after SMOTE
ax = df_rose.plot(kind = 'bar', color = 'blue', label = 'Greater than 113 views', figsize = (15,5))
ax.set_xticklabels(df_smote['Day']) 

#SAVE THE PLOT FIGURE IN CURRENT DIRECTORY
fig = ax.get_figure()
fig.savefig('ROSE_Oversampled_plot')

#New number of 1's (minority class)
n1 = df_rose['ROSE_Oversampled_views'].value_counts()[1]
print('New status of minority class (views>113): ', n1*100/len(df_rose), '%')
