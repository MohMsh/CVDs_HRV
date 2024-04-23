from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import make_scorer
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from numpy import mean, std
import joblib

import warnings

#=====================================================================================
#Turn off all warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

def ApplyAdaBoost(DataFrame):

    print("---------------------------------\n\nImplementing AdaBoost on SHAREEDB\n\n---------------------------------")
    
    # Splitting Data Between Features & Results
    X = DataFrame.iloc[:, 2:].values
    Y = DataFrame.iloc[:, 1].values

    # Transform the dataset using SMOTE
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)

    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.21, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    # define base model
    base = DecisionTreeClassifier(max_depth=1)

    # Define the model
    model = AdaBoostClassifier(
        n_estimators=200,
        base_estimator=base,
        learning_rate=1)

    #Fit the model on the Training dataset
    model.fit(X_Train, Y_Train)

    # make a single prediction
    predictions = model.predict(X_Test)

    '''
    i=0
    result = False
    print("\n\n-----------------------------------------------------")
    for x in Y_Test:
        if(x==predictions[i]):
            result = True
        else:
            result = False
        print("Initial: ", x, "==> Predicted: ", predictions[i], " (", result,")")
        i+=1
    '''
    
    print("\n\nPerformance Metrics: \n-----------------------------------------")
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy: ", metrics.accuracy_score(Y_Test, predictions))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision: ", metrics.precision_score(Y_Test, predictions))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall: ", metrics.recall_score(Y_Test, predictions))

    # Model F1 Score: F1 Score might be a better measure to use 
    # if we need to seek a balance between Precision and Recall
    print("F1 Score: ", metrics.f1_score(Y_Test, predictions))

    # Model Specificity: a model's ability to predict true negatives of each available category
    print("Specificity: ", metrics.recall_score(Y_Test, predictions, pos_label=0))

    # Model Negative Predictive Value (NPV): 
    print("Negative Predictive Value (NPV): ", metrics.precision_score(Y_Test, predictions, pos_label=0))
    print("-----------------------------------------\n")

    #Ask wether to save the model or not
    SaveModel = input("Do you want to save the model? (Y/N):")
    if(SaveModel == "Y"):
        #==========================================================================================
        #save the model to be used later
        filename = 'SavedModels/SVM.sav'
        joblib.dump(ClassifierLinear, filename)
    
        # load the model from disk
        #loaded_model = joblib.load(filename)
        #result = loaded_model#.score(X_Test, Y_Test)
        #print("the saved model", result)
        #==========================================================================================
        
    #Plot the ROC Curve
    predictions = model.predict_proba(X_Test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_Test,  predictions)
    auc = metrics.roc_auc_score(Y_Test, predictions)
    plt.plot(fpr,tpr,label="data, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()


    # Code for Repeated_K-Fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    print("\n\nThe performance metrics after K-Fold Cross Validation")
    # evaluate Accuracy
    FinalAccuracy = cross_val_score(model, X_Train, Y_Train, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    print('Accuracy: %.3f (%.3f)' % (mean(FinalAccuracy), std(FinalAccuracy)))

    # evaluate Precision
    FinalPrecission = cross_val_score(model, X_Train, Y_Train, scoring='precision', cv=cv, n_jobs=-1, verbose=0)
    print('Precision: %.3f (%.3f)' % (mean(FinalPrecission), std(FinalPrecission)))

    # evaluate Recall
    FinalRecall = cross_val_score(model, X_Train, Y_Train, scoring='recall', cv=cv, n_jobs=-1, verbose=0)
    print('Recall: %.3f (%.3f)' % (mean(FinalRecall), std(FinalRecall)))

    # evaluate F1 Score
    FOneScore = cross_val_score(model, X_Train, Y_Train, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    print('F1 Score: %.3f (%.3f)' % (mean(FOneScore), std(FOneScore)))

    # evaluate Specificity
    Specificity = cross_val_score(model, X_Train, Y_Train, scoring=make_scorer(metrics.recall_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Specificity: %.3f (%.3f)' % (mean(Specificity), std(Specificity)))

    # evaluate Negative Predictive Value (NPV)
    NPV = cross_val_score(model, X_Train, Y_Train, scoring=make_scorer(metrics.precision_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Negative Predictive Value (NPV): %.3f (%.3f)' % (mean(NPV), std(NPV)))
