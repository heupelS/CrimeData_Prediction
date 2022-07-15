# %%
from time import time
from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import graphviz
import datetime

# %% Cleaning

def clean_data():
    crimeData = pd.DataFrame(pd.read_csv(data_path))

    crimeData = crimeData.dropna(subset=['AREA', 'TIME.OCC', 'CrmCd.Desc'])
    crimeData['date2'] = pd.to_datetime((crimeData['DATE.OCC']))
    crimeData['Year'] = crimeData['date2'].dt.year
    crimeData['Month'] = crimeData['date2'].dt.month
    crimeData['Day'] = crimeData['date2'].dt.day

    time2 = []

    # convert TIME.OCC into readable time format
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
    crimeData['time2'] = pd.to_datetime(crimeData['time2'], format='%H%M')
    crimeData['Hour'] = crimeData['time2'].dt.hour
    crimeData['Minute'] = crimeData['time2'].dt.minute

    crimeData = crimeData.drop(['date2'], axis=1)
    crimeData = crimeData.drop(['time2'], axis=1)

    crimeData.head(5)

    category_numbers = [[110, 111, 113, 230, 231, 235, 236, 237, 250, 251, 434, 435, 436, 437, 622, 623, 624, 625, 626, 627, 647, 763, 812, 813, 860, 870, 910, 920, 922, 928, 930, 952
                         ], [121, 122, 805, 810, 815, 820, 821, 830, 840, 850, 932, 933
                             ], [210, 220, 310, 320, 330, 331, 341, 343, 345, 347, 349, 350, 351, 352, 353, 410, 420, 421, 430, 431, 433, 440, 441, 442, 443, 444, 445, 446, 450, 451, 452, 470, 471, 473, 474, 475, 480, 485, 487, 510, 520, 653, 654, 668, 670, 940, 950, 951
                                 ], [354, 649, 651, 652, 660, 662, 664, 666
                                     ], [648, 740, 745, 924
                                         ], [753, 756, 761, 931
                                             ], [865
                                                 ], [439, 900, 901, 902, 903
                                                     ], [755, 882, 884, 886, 944
                                                         ], [942
                                                             ], [948
                                                                 ], [438, 890, 997
                                                                     ], [432, 661, 762, 806, 845, 880, 888, 943, 946, 949, 954, 956
                                                                         ]]

    category_names = ['Offences against the Person', 'Sexual Offences', 'Offences under the Theft and Fraud Acts','Forgery, Personation and Cheating','Criminal Damage and Kindred Offences','Firearms and Offensive Weapons',
    'Harmful or Dangerous Drugs','Offences against Public Justice','Public Order Offences','Business, Revenue and Customs Offences',  'Offences Relating to Marriage, Public Nuisance and Obscene or Indecent Publications','Motor Vehicle Offences'
    ,'Other']

    new_categories = []
    for number in crimeData['Crm.Cd']:
        for i in range(13):
            if number in category_numbers[i]:
                new_categories.append(category_names[i])
                break
    
    crimeData['Categories'] = new_categories
    
    print("Total number of crimes in the dataset: {}".format(len(crimeData)))
    
    crimeData['AREA.NAME'] = pd.factorize(crimeData["AREA.NAME"])[0]
    return crimeData

# %% features and targets


def features_target():
    features = ['Hour']
    target = 'Categories'
    category_names = ['Offences against the Person', 'Sexual Offences', 'Offences under the Theft and Fraud Acts','Forgery, Personation and Cheating','Criminal Damage and Kindred Offences','Firearms and Offensive Weapons',
    'Harmful or Dangerous Drugs','Offences against Public Justice','Public Order Offences','Business, Revenue and Customs Offences',  'Offences Relating to Marriage, Public Nuisance and Obscene or Indecent Publications','Motor Vehicle Offences'
    ,'Other']
    return features, target, category_names

# %% dividing into train and test


def train_test(crimeData):
    crimeData_train, crimeData_test = train_test_split(
        crimeData, test_size=0.33, random_state=10)
    return crimeData_train, crimeData_test

# %% Decision tree modeling


def decision_tree(crimeData_train, crimeData_test, features, target):
    clf = tree.DecisionTreeClassifier(max_depth=4)
    cl_fit = clf.fit(crimeData_train[features], crimeData_train[target])
    print("Model Accuracy:")
    print(cl_fit.score(crimeData_test[features], crimeData_test[target]))
    return cl_fit

# %% visualization

def visualize(crimeData, features, category_names, cl_fit):
    listOfClassNames = category_names
    dot_data = tree.export_graphviz(cl_fit, out_file=None, feature_names=features,
                                    class_names=listOfClassNames, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_pdf("decTree_crimeData.pdf")

# %


def get_all_crimetypes(crimeData):
    return crimeData['CrmCd.Desc'].unique()


# %%


if __name__ == "__main__":

    data_path = "Data/Crimes_2012-2016.csv"
    crimeData = clean_data()
    features, target, category_names = features_target()
    crimeData_train, crimeData_test = train_test(crimeData)
    cl_fit = decision_tree(crimeData_train, crimeData_test, features, target)
    visualize(crimeData, features, category_names, cl_fit)


# %%
