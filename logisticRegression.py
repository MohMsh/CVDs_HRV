import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from numpy import mean
from numpy import std
from sklearn.utils import class_weight
import warnings
import joblib

#%matplotlib inline

#Ignore all warnings
#-------------------
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

def ApplyLogisticRegression(DataFrame):

    print("---------------------------------\n\nImplementing Logistic Regression on SHAREEDB\n\n---------------------------------")

    #split dataset in features and target variable
    X = DataFrame.iloc[:, 2:].values
    Y = DataFrame.iloc[:, 1].values

    # Transform the dataset using SMOTE
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)

    # split X and y into training and testing sets
    X_Train,X_test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.29,random_state=42)
    
    # Normalizing continuous variables
    scaler = StandardScaler()
    scaler.fit(X_Train)
    X_Train = scaler.transform(X_Train)
    X_test = scaler.transform(X_test)
    
    class_weights = class_weight.compute_class_weight(
                             class_weight = "balanced",
                             classes = np.unique(Y_Train),
                             y = Y_Train                                                    
                             )
    class_weights = dict(zip(np.unique(Y_Train), class_weights))

    # instantiate the model (using the default parameters)
    # The compatibility between solver and penalty is:
    #'newton-cg' - [‘l2’, ‘none’]
    #'lbfgs' - [‘l2’, ‘none’]
    #'liblinear' - [‘l1’, ‘l2’]
    #'sag' - [‘l2’, ‘none’]
    #'saga' - [‘elasticnet’, ‘l1’, ‘l2’, ‘none’]
    logreg = LogisticRegression(penalty = 'none', C =3.1, solver = 'newton-cg', verbose = 0, class_weight=class_weights)

    # fit the model with data
    logreg.fit(X_Train,Y_Train)

    #Predict the Y_test using the model
    Y_Pred=logreg.predict(X_test)

    #get the confusion matrix
    cm = metrics.confusion_matrix(Y_Test, Y_Pred)
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
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred))
    
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(Y_Test, Y_Pred))
  
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(Y_Test, Y_Pred))
    
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
    
    #Plot the ROC Curve
    Y_Pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(Y_Test,  Y_Pred_proba)
    auc = metrics.roc_auc_score(Y_Test, Y_Pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

    # Code for Repeated_K-Fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    print("The performance metrics after K-Fold Cross Validation")
    # evaluate Accuracy
    FinalAccuracy = cross_val_score(logreg, X_Train, Y_Train, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    print('Accuracy: %.3f (%.3f)' % (mean(FinalAccuracy), std(FinalAccuracy)))
    
    # evaluate Precision
    FinalPrecission = cross_val_score(logreg, X_Train, Y_Train, scoring='precision', cv=cv, n_jobs=-1, verbose=0)
    print('Precision: %.3f (%.3f)' % (mean(FinalPrecission), std(FinalPrecission)))
    
    # evaluate Recall
    FinalRecall = cross_val_score(logreg, X_Train, Y_Train, scoring='recall', cv=cv, n_jobs=-1, verbose=0)
    print('Recall: %.3f (%.3f)' % (mean(FinalRecall), std(FinalRecall)))
    
    # evaluate F1 Score
    FOneScore = cross_val_score(logreg, X_Train, Y_Train, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    print('F1 Score: %.3f (%.3f)' % (mean(FOneScore), std(FOneScore)))
    
    # evaluate Specificity
    Specificity = cross_val_score(logreg, X_Train, Y_Train, scoring=make_scorer(metrics.recall_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Specificity: %.3f (%.3f)' % (mean(Specificity), std(Specificity)))
    
    # evaluate Negative Predictive Value (NPV)
    NPV = cross_val_score(logreg, X_Train, Y_Train, scoring=make_scorer(metrics.precision_score, pos_label=0), cv=cv, n_jobs=-1, verbose=0)
    print('Negative Predictive Value (NPV): %.3f (%.3f)' % (mean(NPV), std(NPV)))
    
    
    #Ask wether to save the model or not
    SaveModel = input("Do you want to save the model? (Y/N):")
    if(SaveModel == "Y"):
        #==========================================================================================
        #save the model to be used later
        filename = 'SavedModels/LR.sav'
        joblib.dump(logreg, filename)
    
        # load the model from disk
        #loaded_model = joblib.load(filename)
        #result = loaded_model#.score(X_Test, Y_Test)
        #print("the saved model", result)
        #==========================================================================================