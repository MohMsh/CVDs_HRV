import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, make_scorer
from sklearn import metrics
from sklearn import svm
from time import time
import seaborn as sns
from imblearn.over_sampling import SMOTE
from numpy import mean
from numpy import std
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, auc
import joblib

def ApplySVM(DataFrame):

    print("---------------------------------\n\nImplementing Support Vector Machines on SHAREEDB\n\n---------------------------------")

    # Splitting Data Between Features & Results
    X = DataFrame.iloc[:, 2:].values
    Y = DataFrame.iloc[:, 1].values

    print(X)
    print(Y)
    
    # Transform the dataset using SMOTE
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)
    

    # Splitting the dataset into the Training set and Test set
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)

    class_weights = class_weight.compute_class_weight(
                             class_weight = "balanced",
                             classes = np.unique(Y_Train),
                             y = Y_Train                                                    
                             )
    class_weights = dict(zip(np.unique(Y_Train), class_weights))
    
    # Fitting the classifier into the Training set
    # The 'balanced class_weight is good for heuristic balance
    ClassifierLinear = SVC(kernel='rbf', C=2.66, gamma=0.141, class_weight=class_weights,probability=True)
    
    start = time()
    ClassifierLinear.fit(X_Train, Y_Train)
   
    # Predicting the test set results
    Y_Pred = ClassifierLinear.predict(X_Test)
    
    # Making the Confusion Matrix 
    cm = confusion_matrix(Y_Test, Y_Pred)
    #calculate TP, TN, FP and FN
    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Linear Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy: ", metrics.accuracy_score(Y_Test, Y_Pred))
    
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision: ", metrics.precision_score(Y_Test, Y_Pred))
    
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall: ", metrics.recall_score(Y_Test, Y_Pred))
    
    # Model F1 Score: F1 Score might be a better measure to use 
    # if we need to seek a balance between Precision and Recall
    print("F1 Score: ", metrics.f1_score(Y_Test, Y_Pred))
    
    # Model Specificity: a model's ability to predict true negatives of each available category
    print("Specificity: ", metrics.recall_score(Y_Test, Y_Pred, pos_label=0))
    
    # Model Negative Predictive Value (NPV): 
    print("Negative Predictive Value (NPV): ", metrics.precision_score(Y_Test, Y_Pred, pos_label=0))
    
    # print classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Classification error : {0:0.4f}'.format(classification_error))
    print("===============================================================================\n")
    print(classification_report(Y_Test, Y_Pred))
    print("===============================================================================\n")
    #==========================================================================================
    
    #Ask wether to save the model or not
    SaveModel = input("---------------------------------\n\nDo you want to save the model? (Y/N):")
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
    y_pred_proba = ClassifierLinear.predict_proba(X_Test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_Test,  y_pred_proba)
    auc = metrics.roc_auc_score(Y_Test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

    
    # Code for Repeated_K-Fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
 
    print("The performance metrics after K-Fold Cross Validation")
    # evaluate Accuracy
    FinalAccuracy = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    print('Accuracy: %.3f (%.3f)' % (mean(FinalAccuracy), std(FinalAccuracy)))
    
    # evaluate Precision
    FinalPrecission = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring='precision', cv=cv, n_jobs=-1, verbose=0)
    print('Precision: %.3f (%.3f)' % (mean(FinalPrecission), std(FinalPrecission)))
    
    # evaluate Recall
    FinalRecall = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring='recall', cv=cv, n_jobs=-1, verbose=0)
    print('Recall: %.3f (%.3f)' % (mean(FinalRecall), std(FinalRecall)))
    
    # evaluate F1 Score
    FOneScore = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    print('F1 Score: %.3f (%.3f)' % (mean(FOneScore), std(FOneScore)))
    
    # evaluate Specificity
    Specificity = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring=make_scorer(metrics.recall_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Specificity: %.3f (%.3f)' % (mean(Specificity), std(Specificity)))
    
    # evaluate Negative Predictive Value (NPV)
    NPV = cross_val_score(ClassifierLinear, X_Train, Y_Train, scoring=make_scorer(metrics.precision_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Negative Predictive Value (NPV): %.3f (%.3f)' % (mean(NPV), std(NPV)))