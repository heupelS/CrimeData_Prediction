# %%
from time import time
from unicodedata import category
from xmlrpc.client import Boolean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
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

    daytime = []

    for hour in crimeData['Hour']:
        if hour < 3 or hour >= 21:
            daytime.append('Night')
        elif hour < 9:
            daytime.append('Morning')
        elif hour < 15:
            daytime.append('Midday')
        else:
            daytime.append('Evening')

    crimeData['Daytime'] = daytime
    crimeData['daytime_fact'] = pd.factorize(crimeData['Daytime'])[0]

    daytime_selector = [[] for i in range(4)]

    for row in crimeData['daytime_fact']:
        for i in range(4):
            if i == row:
                daytime_selector[i].append(True)
            else:
                daytime_selector[i].append(False)

    global daytime_selector_names
    daytime_selector_names = []
    
    for i in range(4):
        daytime_selector_names.append('daytime_selector_'+str(i))
        crimeData['daytime_selector_'+str(i)] = daytime_selector[i]

    season = []

    for month in crimeData['Month']:
        if month < 3 or month == 12:
            season.append('Winter')
        elif month < 6:
            season.append('Spring')
        elif month < 9:
            season.append('Summer')
        else:
            season.append('Fall')

    crimeData['Season'] = season
    crimeData['season_fact'] = pd.factorize(crimeData['Season'])[0]

    season_selector = [[] for i in range(4)]

    for row in crimeData['season_fact']:
        for i in range(4):
            if i == row:
                season_selector[i].append(True)
            else:
                season_selector[i].append(False)

    global season_selector_names
    season_selector_names = []
    
    for i in range(4):
        season_selector_names.append('season_selector_'+str(i))
        crimeData['season_selector_'+str(i)] = season_selector[i]


    category_numbers = [[510,480,520,487
                         ], [330,410
                             ], [310,320
                                 ], [440,442
                                     ], [420,331
                                         ], [210,220,341,668,343,350
                                             ], [624,860
                                                 ], [230,236
                                                     ], [626,930
                                                         ], [648,740,745,924
                                                             ], [354,649,651,652,660,662,664,666
                                                                 ], [438, 890, 997
                                                                     ], [121,122,805,810,815,820,821,830,840,850,932,933,753,756,761,931,439,900,901,902,903
                                                                         ], []]

    category_names = ['Vehicle Theft','Burglary from Vehicle','Burglary','Petty Theft','Theft From Vehicle','Robbery and Grand Theft',
                        'Battery','Aggravated Assault','Spousal Abuse and Threats','Criminal Damage and Kindred Offences',
                        'Forgery, Personation and Cheating','Motor Vehicle Offences','Sex crimes, firearms and public justice','Other']

    new_categories = []
    for number in crimeData['Crm.Cd']:
        for i in range(14):
            if i == 13:
                new_categories.append(category_names[i])
                break
            if number in category_numbers[i]:
                new_categories.append(category_names[i])
                break

        
    crimeData['Categories'] = new_categories
    crimeData['cat_fact'] = pd.factorize(crimeData["Categories"])[0]

    area_selector = [[] for i in range(21)]

    for row in crimeData['AREA']:
        for i in range(1,22):
            if i == row:
                area_selector[i-1].append(True)
            else:
                area_selector[i-1].append(False)

    global area_selector_names
    area_selector_names = []

    for i in range(1,22):
        area_selector_names.append('area_selector_'+str(i))
        crimeData['area_selector_'+str(i)] = area_selector[i-1]
    
    print("Total number of crimes in the dataset: {}".format(len(crimeData)))
    
    return crimeData

# %% features and targets


def features_target():
    features = area_selector_names + daytime_selector_names + season_selector_names
    target = 'Categories'
    category_names = ['Vehicle Theft','Burglary from Vehicle','Burglary','Petty Theft','Theft From Vehicle','Robbery and Grand Theft',
                        'Battery','Aggravated Assault','Spousal Abuse and Threats','Criminal Damage and Kindred Offences',
                        'Forgery, Personation and Cheating','Motor Vehicle Offences','Sex crimes, firearms and public justice','Other']
    return features, target, category_names

# %% dividing into train and test


def train_test(crimeData):
    crimeData_train, crimeData_test = train_test_split(
        crimeData, test_size=0.33, random_state=20)
    return crimeData_train, crimeData_test

# %% Decision tree modeling


def decision_tree(crimeData_train, crimeData_test, features, target):
    clf = tree.DecisionTreeClassifier(criterion ="gini",max_depth=25)
    cl_fit = clf.fit(crimeData_train[features], crimeData_train[target])
    print("Model Accuracy:")
    #cl_fit2 = clf.predict(crimeData_test[features])
    print(cl_fit.score(crimeData_test[features], crimeData_test[target]))
    return cl_fit

# %% visualization

def visualize(crimeData, features, category_names, cl_fit):
    dot_data = tree.export_graphviz(cl_fit, out_file=None, feature_names=features,
                                    class_names=crimeData[target].unique(), filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_pdf("decTree_crimeData.pdf")

#%% Cross Validation
def cross_validate(crimeData_train):
    scores = cross_val_score(cl_fit, crimeData_train[features], crimeData_train[target], cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)"
    % (scores.mean(), scores.std() ))



#%% neural network
def neural_network(features, target, crimeData_train, crimeData_test):
    nn_model = MLPClassifier(solver='adam', 
                         alpha=1e-7,
                         hidden_layer_sizes=(40,), 
                         random_state=1,
                         max_iter=1000                         
                        )

    # Model Training
    nn_model.fit(crimeData_train[features], crimeData_train[target])

    # Prediction
    result = nn_model.predict(crimeData_test[features]) 
    print(result)
    print(nn_model.score(crimeData_test[features], crimeData_test[target]))

#%%
def get_all_crimetypes(crimeData):
    return crimeData['CrmCd.Desc'].unique()


#%% majority
def majority_classifier(features, crimeData_train, crimeData_test, target):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(crimeData_train[features], crimeData_train[target])
    DummyClassifier(strategy='most_frequent')
    dummy_clf.predict(crimeData_test[features])
    print(dummy_clf.score(crimeData_test[features], crimeData_test[target]))



# %%


if __name__ == "__main__":

    data_path = "Data/Crimes_2012-2016.csv"
    crimeData = clean_data()
    features, target, category_names = features_target()
    crimeData_train, crimeData_test = train_test(crimeData)
    cl_fit = decision_tree(crimeData_train, crimeData_test, features, target)
    visualize(crimeData,features, category_names, cl_fit)
    majority_classifier(features, crimeData_train, crimeData_test, target)

# %%
