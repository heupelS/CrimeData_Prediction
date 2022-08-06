# %%
import string
from time import time
from unicodedata import category
from xmlrpc.client import Boolean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from pyrsistent import s
from sklearn import datasets
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import pydotplus
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import folium
from sklearn.metrics import plot_confusion_matrix
from folium import plugins
from folium.plugins import HeatMap
import seaborn as sns
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
    
    
    
    # Location in oridnal umwandeln
    le = preprocessing.LabelEncoder()
    le.fit(list(set(crimeData['LOCATION'])))
    crimeData['LOCATION'] = le.transform(crimeData['LOCATION'])

    # X und Y  Koordinate erstellen
    crimeData = crimeData.dropna(subset=['Location.1'])
    crimeData[['Y','X']] = crimeData['Location.1'].str.split(',', expand=True)
    crimeData['Y'] = pd.to_numeric(crimeData['Y'].str[1:])
    crimeData = crimeData[crimeData['Y'] != 0.0]
    crimeData['X'] = pd.to_numeric(crimeData['X'].str[:-1])
    crimeData = crimeData[crimeData['X'] != 0.0]

    print("Total number of crimes in the dataset: {}".format(len(crimeData)))
    return crimeData

# %% features and targets


def features_target():
    features = ['X','Y','Hour','season_fact','daytime_fact'] + area_selector_names + season_selector_names + daytime_selector_names
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
    clf = tree.DecisionTreeClassifier(criterion ="gini",max_depth=15)
    cl_fit = clf.fit(crimeData_train[features], crimeData_train[target])
    print("Model Accuracy:")
    result = clf.predict(crimeData_test[features])
    #print(cl_fit.score(crimeData_test[features], crimeData_test[target]))
    print(cl_fit.score(crimeData_test[features], crimeData_test[target]))
    print(cl_fit.score(crimeData_train[features], crimeData_train[target]))
    return result

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
                         alpha=1e-5,
                         hidden_layer_sizes=(20,),
                         learning_rate='adaptive', 
                         random_state=1,
                         max_iter=20                         
                        )

    # Model Training
    nn_model.fit(crimeData_train[features], crimeData_train[target])

    # Prediction
    result = nn_model.predict(crimeData_test[features])
    return result

#%% random functions
def get_all_crimetypes(crimeData):
    return crimeData['CrmCd.Desc'].unique()

#%% majority classifier
def majority_classifier(features, crimeData_train, crimeData_test, target):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(crimeData_train[features], crimeData_train[target])
    DummyClassifier(strategy='most_frequent')
    result = dummy_clf.predict(crimeData_test[features])
    print(dummy_clf.score(crimeData_test[features], crimeData_test[target]))
    return result

#%% logistic regression
def logistic_regression(features, crimeData_train, crimeData_test, target):
    clf = LogisticRegression(class_weight = 'balanced',max_iter = 50, random_state=0)
    clf = clf.fit(crimeData_train[features], crimeData_train[target])
    result = clf.predict(crimeData_test[features])
    return result


#%%
def evaluate(model_name: string, Y_test, result,category_names):
    accuracy = accuracy_score(Y_test, result)
    recall = recall_score(Y_test, result, average="weighted")
    precision = precision_score(Y_test, result, average="weighted")
    f1 = f1_score(Y_test, result, average='micro')
    confusion_m = confusion_matrix(Y_test, result)

    print(f'------------- {model_name} -------------')
    print("Accuracy    : ", accuracy)
    print("Recall      : ", recall)
    print("Precision   : ", precision)
    print("F1 Score    : ", f1)
    print("Confusion Matrix: ")
    #confused confusion matrix :/
    # %%
    fig, ax = plt.subplots(figsize=(10, 10))
    #decisiontreeresult muss der classifier sein nicht ergebnisse
    #plot_confusion_matrix(decision_tree_result, crimeData_test[features], crimeData_test[target],xticks_rotation='vertical', ax=ax)
    """ df_cm = pd.DataFrame(confusion_m, index = category_names[::-1], columns = category_names)
    plt.figure(figsize = (14,10))
    hm = sns.heatmap(df_cm, annot=True,cmap="OrRd", fmt='g')
    plt.title("Confusion Matrix")
    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show() """
#%%
def visualize_categories_vs_predictions(model_name:string,crimeData_test,target,predictions):

    predictions = pd.DataFrame(predictions, columns=[target])
    dfs = {'crimeData_test': crimeData_test, model_name:predictions} # create dictionary for df names to print them
    for key in dfs:
        dataframe = dfs[key]
        print (f'--------- {key} ---------' )
        plt.figure(figsize=(14,10))
        plt.title('Anzahl der Verbrechen nach Kategorie')
        plt.ylabel('Verbrechen Kategorie')
        plt.xlabel('Anzahl der Verbrechen')

        dataframe.groupby(dataframe[target]).size().sort_values(ascending=True).plot(kind='barh',cmap="plasma", ylabel= 'Straftat Kategorie')


#%%
""" def max_test():
    crimeData.drop(crimeData.index[crimeData['AREA.NAME'] == 'Olympic'], inplace = True)
    crimeData.drop(crimeData.index[crimeData['AREA.NAME'] == 'Newton'], inplace = True)
    crimeData.drop(crimeData.index[crimeData['AREA.NAME'] == 'Topanga'], inplace = True)

    districts = ['Hollywood', 'Southeast', 'Harbor', '77th Street', 'Central',
       'Van Nuys', 'Foothill', 'Northeast',
       'Southwest', 'West LA', 'Mission', 'West Valley', 'Rampart',
       'Hollenbeck', 'Devonshire', 'Pacific', 'N Hollywood', 'Wilshire']

    district_area = pd.Series([17.2,10.2,27,11.9,4.5,30,46.13,29,13.11,65.14,25.1,33.5,5.54,15.2,48.31,25.74,25,13.97],index=districts)

    district_population = pd.Series([300000,150000,171000,175000,40000,325000,182214,250000,165000,228000,225849,196840,164961,200000,219136,200000,220000,251000],index=districts)

    global perArea
    global perPopulation
    perArea = pd.Series([],index=[],dtype=np.float64)
    perPopulation = pd.Series([],index=[],dtype=np.float64)

    for district in districts:
        occurences = crimeData['AREA.NAME'].value_counts(sort=False)[district]
        area = district_area[district]
        population = district_population[district]

        tmp1 = pd.Series([occurences/area], index=[district])
        tmp2 = pd.Series([occurences/population], index=[district])

        perArea = perArea.append(tmp1)
        perPopulation = perPopulation.append(tmp2)

    #perAreaVis(perArea=perArea)
    #perPopVis(perPopulation=perPopulation)


def perAreaVis(perArea):
    perArea.plot(kind="bar",figsize=(9,8), cmap = 'coolwarm')
    _=plt.title('Crime pro Area', fontsize=20)

def perPopVis(perPopulation):
    perPopulation.plot(kind="bar",figsize=(9,8), cmap = 'coolwarm')
    _=plt.title('Crime pro Citizen', fontsize=20) """

#%%
""" def visMap():
    coordinates = []
    for crdnts in crimeData['Location.1']:
        tup = tuple(map(float, str(crdnts).replace('(','').replace(')','').replace(' ','').split(',')))
        coordinates.append(tup)
    crimeData['coordinates'] = coordinates """

# %%


if __name__ == "__main__":

    # initialisierung
    data_path = "Data/Crimes_2012-2016.csv"
    crimeData = clean_data()
    features, target, category_names = features_target()
    crimeData_train, crimeData_test = train_test(crimeData)

    #majority classifier
    majority_classifier_result = majority_classifier(features, crimeData_train, crimeData_test, target)
    evaluate('majority classifier',crimeData_test[target],majority_classifier_result,category_names)
    #visualize_categories_vs_predictions('majority classifier',crimeData_test,target,majority_classifier_result)
    
    #decision tree
    decision_tree_result = decision_tree(crimeData_train, crimeData_test, features, target)
    evaluate('decision tree',crimeData_test[target],decision_tree_result, category_names)
    #visualize_categories_vs_predictions('decision tree',crimeData_test,target,decision_tree_result)
    
    # dec tree to large to create png of entire tree
    # visualize(crimeData,features, category_names, cl_fit)
    
    # deep neural network
    dnn_result = neural_network(features, target, crimeData_train, crimeData_test)
    evaluate('Deep Neural Network',crimeData_test[target],dnn_result, category_names)
    #visualize_categories_vs_predictions('Deep Neural Network',crimeData_test,target,dnn_result)

    # logistic regression
    log_regression_result = neural_network(features, target, crimeData_train, crimeData_test)
    evaluate('logistic regression',crimeData_test[target],log_regression_result, category_names)
    #visualize_categories_vs_predictions('logistic regression',crimeData_test,target,log_regression_result)
    
# %%
