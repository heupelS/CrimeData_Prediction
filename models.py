# %%
############ installations ############
# pip install -U scikit-learn
# pip install folium
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install numpy

import string
from time import time
from unicodedata import category
from unittest import result
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
from sklearn.neighbors import KNeighborsClassifier
import pydotplus
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import folium
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_validate
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

    #crimeData.drop(crimeData.index[crimeData['Crm.Cd'] == 997], inplace = True)


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

    homicide = ('homicide',[110,111])
    simple_assault_wo_family = ('simple_assault_wo_family',[624,623,622,625])
    aggravated_assault_wo_family = ('aggravated_assault_wo_family',[230,231])
    family_violence_simple = ('family_violence_simple',[626,627])
    family_violence_aggravated = ('family_violence_aggravated',[235,236])
    threats = ('threats',[930,928])
    theft_wo_identity = ('theft_wo_identity',[440,341,420,331,350,441,421,450,474,473])
    identity_theft = ('identity_theft',[354])
    stolen_vehicle = ('stolen_vehicle',[510])
    robbery = ('robbery',[210,220])
    burglary = ('burglary',[310,320,330,410])
    shoplifting = ('shoplifting',[442,343])
    vandalism = ('vandalism',[740,745])
    violent_sex_crimes = ('violent_sex_crimes',[121,122,860,821,815])
    motor_vehicle_offenses = ('motor_vehicle_offenses',[997,438])

    violent_crimes = ('violent_crimes',
                        homicide[1]+
                        simple_assault_wo_family[1]+
                        aggravated_assault_wo_family[1]+
                        family_violence_simple[1]+
                        family_violence_aggravated[1]+
                        violent_sex_crimes[1]+
                        robbery[1])
    all_theft = ('all_theft',theft_wo_identity[1]+identity_theft[1])
    burglary_and_stolen_vehicle = ('burglary_and_stolen_vehicle',burglary[1]+stolen_vehicle[1])

    global category_tuples
    category_tuples = [violent_crimes,
                        #theft_wo_identity,
                        #identity_theft,
                            all_theft,
                        threats,
                        burglary,
                        stolen_vehicle,
                            #burglary_and_stolen_vehicle,
                        shoplifting,
                        vandalism,
                        motor_vehicle_offenses]

    new_categories = []
    for number in crimeData['Crm.Cd']:
        for i in range(len(category_tuples)+1):
            if i == len(category_tuples):
                new_categories.append('other')
                break
            if number in category_tuples[i][1]:
                new_categories.append(category_tuples[i][0])
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
    features = ['X','Y','season_fact','daytime_fact'] + area_selector_names + season_selector_names + daytime_selector_names
    target = 'Categories'
    category_names = []
    for item in category_tuples:
        category_names.append(item[0])
    category_names.append('other')
    return features, target, category_names

# %% dividing into train and test


def train_test(crimeData):
    crimeData_train, crimeData_test = train_test_split(
        crimeData, test_size=0.2, random_state=20)
    return crimeData_train, crimeData_test

# %% Decision tree modeling


def decision_tree(crimeData_train, crimeData_test, features, target,max_depth):
    clf = tree.DecisionTreeClassifier(criterion ="gini",max_depth=max_depth)
    cl_fit = clf.fit(crimeData_train[features], crimeData_train[target])
    result = clf.predict(crimeData_test[features])
    #print("cross val:")
    #scores = cross_validate(cl_fit,crimeData)
    return result

# %% visualization

def visualize(crimeData, features, category_names, cl_fit):
    dot_data = tree.export_graphviz(cl_fit, out_file=None, feature_names=features,
                                    class_names=crimeData[target].unique(), filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    graph.write_pdf("decTree_crimeData.pdf")

#%% Cross Validation
def cross_validate(clf,crimeData_train):
    scores = cross_val_score(clf, crimeData_train[features], crimeData_train[target], cv=5)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() ))
    return scores



#%% neural network
def neural_network(features, target, crimeData_train, crimeData_test,max_iter):
    nn_model = MLPClassifier(solver='adam', 
                         alpha=1e-5,
                         hidden_layer_sizes=(20,),
                         learning_rate='adaptive', 
                         random_state=1,
                         max_iter=max_iter                         
                        )

    # Model Training
    nn_model.fit(crimeData_train[features], crimeData_train[target])

    # Prediction
    result = nn_model.predict(crimeData_test[features])

    #cross validation
    print("cross val:")
    scores = cross_validate(nn_model,crimeData)

    return result

#%% majority classifier
def majority_classifier(features, crimeData_train, crimeData_test, target):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(crimeData_train[features], crimeData_train[target])
    DummyClassifier(strategy='most_frequent')
    result = dummy_clf.predict(crimeData_test[features])
    print(dummy_clf.score(crimeData_test[features], crimeData_test[target]))
    return result

#%% logistic regression
def logistic_regression(features, crimeData_train, crimeData_test, target,max_iter):
    clf = LogisticRegression(class_weight = 'balanced',max_iter = max_iter, random_state=0)
    clf = clf.fit(crimeData_train[features], crimeData_train[target])
    result = clf.predict(crimeData_test[features])
    
    #cross validation
    print("cross val:")
    scores = cross_validate(clf,crimeData)

    return result

#%%
def knn(features, crimeData_train, crimeData_test, target, n_neighbors):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(crimeData_train[features], crimeData_train[target])
    results = neigh.predict(crimeData_test[features])
    
    #cross validation
    print("cross val:")
    scores = cross_validate(neigh,crimeData)
    return results


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
    #fig, ax = plt.subplots(figsize=(10, 10))
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

#%% plot_fitting_graph_error_vs_complexity
def plot_fitting_graph_error_vs_complexity(model_name,model,par_name,param_range,crimeData_train,target):
    param_range = [1, 5, 10, 15, 20, 25,100]
    train_scores, test_scores = validation_curve(
        model,
        crimeData_train[features],
        crimeData_train[target],
        param_name=par_name,
        param_range=param_range,
        cv = 5,
        scoring="accuracy",
        n_jobs=2,
    )
    train_error = 1 - np.mean(train_scores, axis=1)
    test_error = 1 - np.mean(test_scores, axis=1)

    plt.title(f"Validation curve for {model_name}")
    plt.xlabel(f"Komplexität ({par_name})")
    plt.ylabel("Error in %")
    plt.plot(param_range,train_error,label = 'Training Score')
    plt.plot(param_range,test_error,label = 'Cross Val. Test Score')
    plt.legend()
    plt.show()

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
    visualize_categories_vs_predictions('majority classifier',crimeData_test,target,majority_classifier_result)
    
    #decision tree
    decision_tree_result = decision_tree(crimeData_train, crimeData_test, features, target,15)
    evaluate('decision tree',crimeData_test[target],decision_tree_result, category_names)
    visualize_categories_vs_predictions('decision tree',crimeData_test,target,decision_tree_result)
    plot_fitting_graph_error_vs_complexity('decision tree',model = tree.DecisionTreeClassifier(criterion ="gini"),
                                            par_name='max_depth',param_range=[1, 5, 10, 15, 20, 25,100],
                                            crimeData_train = crimeData_train,target = target)
    
    # dec tree to large to create png of entire tree
    # visualize(crimeData,features, category_names, cl_fit)

    # knn 
    # this part takes about 13 min to run 
    knn_result = knn(features, crimeData_train, crimeData_test, target,5)
    evaluate('k nearest neighbor',crimeData_test[target],knn_result, category_names)
    visualize_categories_vs_predictions('k nearest neighbor',crimeData_test,target,knn_result)

    # this function takes really long to plot (> 60 min)
    # we only need to run it one time to have our optimal complexity degree (n_neighbors)!
    # comment in the three lines below to run
    #plot_fitting_graph_error_vs_complexity('k nearest neighbor',KNeighborsClassifier(),
    #                                        par_name='n_neighbors',param_range=[1,3,5,7,10],
    #                                        crimeData_train = crimeData_train,target = target)

    # deep neural network
    dnn_result = neural_network(features, target, crimeData_train, crimeData_test,20)
    evaluate('Deep Neural Network',crimeData_test[target],dnn_result, category_names)
    visualize_categories_vs_predictions('Deep Neural Network',crimeData_test,target,dnn_result)
    
    # this function takes really long to plot
    #plot_fitting_graph_error_vs_complexity('Deep Neural Network',model = MLPClassifier(solver='adam', alpha=1e-5,
    #                                        hidden_layer_sizes=(20,),
    #                                        learning_rate='adaptive', random_state=1,                       
    #                                        ),
    #                                        par_name='max_iter',param_range=[5, 10, 15, 20, 25, 100, 250],
    #                                        crimeData_train = crimeData_train,target = target)

    # logistic regression <--- funktioniert nicht
    log_regression_result = logistic_regression(features, crimeData_train, crimeData_test, target,200)
    evaluate('logistic regression',crimeData_test[target],log_regression_result, category_names)
    visualize_categories_vs_predictions('logistic regression',crimeData_test,target,log_regression_result)
    
    # this function takes really long to plot
    #plot_fitting_graph_error_vs_complexity('Deep Neural Network',model = LogisticRegression(class_weight = 'balanced'),
    #                                        par_name='max_iter',param_range=[5, 10, 15, 20, 25, 100, 250],
    #                                        crimeData_train = crimeData_train,target = target)
    
# %%
