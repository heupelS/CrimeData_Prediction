#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import graphviz


#%% Overview
print("Total number of crimes in the dataset: {}".format(len(crimeData)))
crimeData.head()

#%% features and targets

features = ['AREA','Crm.Cd','RD','TIME.OCC']
targets = ['AREA.NAME','LOCATION','DATE.OCC']

#%% dividing into train and test
crimeData_train, crimeData_test = train_test_split(crimeData, test_size=0.33, random_state=10)

#%% Decision tree modeling

clf = tree.DecisionTreeClassifier(max_depth=8)
cl_fit = clf.fit(crimeData_train[features], crimeData_train['AREA.NAME'])
print("Model Accuracy:")
print(cl_fit.score(crimeData_test[features],crimeData_test['AREA.NAME']))

#%% visualization
listOfClassNames = list(set(crimeData['AREA.NAME']))
dot_data = tree.export_graphviz(cl_fit, out_file=None, feature_names=features, class_names= listOfClassNames, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("decTree_iris.pdf")

# %%
if __name__ == "__main__":

    data_path = "Data/Crimes_2012-2016.csv"

    crimeData = pd.read_csv(data_path)


# %%
