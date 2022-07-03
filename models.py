#%%
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrsistent import T
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import graphviz
import datetime


#%% Overview

crimeData = pd.DataFrame(pd.read_csv(data_path))
print("Total number of crimes in the dataset: {}".format(len(crimeData)))


#%% Cleaning

crimeData = crimeData.dropna(subset=['AREA','TIME.OCC','CrmCd.Desc'])
crimeData['date2'] = pd.to_datetime((crimeData['DATE.OCC']))
crimeData['Year'] = crimeData['date2'].dt.year
crimeData['Month'] = crimeData['date2'].dt.month
crimeData['Day'] = crimeData['date2'].dt.day

time2 = []

#convert TIME.OCC into readable time format
time1 = list(crimeData['TIME.OCC'].astype(str))
for t in range(len(time1)):
    if len(time1[t]) == 4:
        continue
    elif len(time1[t]) == 3:
        time1[t] = '0' + time1[t]
    elif len(time1[t]) == 2:
        time1[t] = '00' + time1[t]
    else:
        time1[t] = '000' + time1[t]
    

crimeData['time2'] = time1
crimeData['time2'] = pd.to_datetime(crimeData['time2'],format='%H%M')
crimeData['Hour'] = crimeData['time2'].dt.hour
crimeData['Minute'] = crimeData['time2'].dt.minute

crimeData = crimeData.drop(['date2'], axis= 1)
crimeData = crimeData.drop(['time2'], axis= 1)

crimeData.head(5)

print("Total number of crimes in the dataset: {}".format(len(crimeData)))
crimeData.head()

#%% features and targets

features = ['AREA','Year','Month','Day','Hour','Minute']
targets = ['AREA.NAME','LOCATION','DATE.OCC']

#%% dividing into train and test
crimeData_train, crimeData_test = train_test_split(crimeData, test_size=0.33, random_state=10)

#%% Decision tree modeling

clf = tree.DecisionTreeClassifier(max_depth=5)
cl_fit = clf.fit(crimeData_train[features], crimeData_train['CrmCd.Desc'])
print("Model Accuracy:")
print(cl_fit.score(crimeData_test[features],crimeData_test['CrmCd.Desc']))

#%% visualization
listOfClassNames = list(set(crimeData['CrmCd.Desc']))
dot_data = tree.export_graphviz(cl_fit, out_file=None, feature_names=features, class_names= listOfClassNames, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("decTree_iris.pdf")

# %%
if __name__ == "__main__":

    data_path = "Data/Crimes_2012-2016.csv"


# %%
