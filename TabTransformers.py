import pandas as pd
import numpy as np
import sklearn
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import TabTransformerConfig
from pyradox_tabular.nn import TabTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from numpy import mean, std
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
import matplotlib.pyplot as plt
import joblib
import warnings


#Turn off all warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

def ApplyTabTransformers(DataFrame):

    print("---------------------------------\n\n Implementing TabTransformers on SHAREEDB\n\n---------------------------------")

    le = LabelEncoder()
    oe = OneHotEncoder()

    # Splitting Data Between Features & Results
    X = DataFrame.iloc[:, 2:]
    Y = DataFrame.iloc[:, 1]

    # Transform the dataset using SMOTE
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)

    numerical_features = X.columns

    # Splitting the dataset into the Training set and Test set
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.28, random_state=42)

    # Feature Scaling
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_valid = sc_X.transform(x_valid)

    #Data Transformation
    y_train = le.fit_transform(y_train)
    y_valid = le.fit_transform(y_valid)

    unique, counts = np.unique(y_train, return_counts=True)
    unique, counts = np.unique(y_valid, return_counts=True)

    y_train = oe.fit_transform(y_train.reshape(-1, 1))
    y_train = y_train.toarray()
    y_train =pd.DataFrame(y_train)

    y_valid = oe.fit_transform(y_valid.reshape(-1, 1))
    y_valid = y_valid.toarray()
    y_valid =pd.DataFrame(y_valid)

    RobustScaler_transformer = RobustScaler().fit(x_train)
    X_trainRobustScaler = RobustScaler_transformer.transform(x_train)
    X_validRobustScaler = RobustScaler_transformer.transform(x_valid)


    x_train =pd.DataFrame(
        X_trainRobustScaler,
        columns=list(numerical_features))


    x_valid =pd.DataFrame(
        X_validRobustScaler,
        columns=list(numerical_features))

    #define data configuration to use it in model
    data_config = DataConfig(
        numeric_feature_names= ['a','b','c','d','e','f','g','h','i','j','k','l','m',
                                'n','o','p','q','r','s','t','u','v','w','x','y','z'],
        categorical_features_with_vocabulary={},
    )


    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)
    data_test = DataLoader.from_df(x_valid, batch_size=1024)

    #Configure the model
    model_config = TabTransformerConfig(num_outputs=2,
                                        out_activation='sigmoid',
                                        num_transformer_blocks=1024,
                                        num_heads=1024,  # Number of attention heads.
                                        dropout_rate=0.1, 
                                        mlp_hidden_units_factors=[1024, 512] # MLP hidden layer units, as factors of the number of inputs.
                                        )

    #Build the model
    model = TabTransformer.from_config(data_config,
                                       model_config,
                                       name="tab_transformer")

    #Define the optimizer
    opt = keras.optimizers.SGD(learning_rate=0.05)

    #Compile the model
    model.compile(optimizer=opt,
                  loss="binary_crossentropy")


    #train the model
    fitted_model = model.fit(data_train,
                             validation_data=data_valid,
                             epochs=2000)

    #predict the model
    test_preds = model.predict(data_test)
    test_preds_decoded = oe.inverse_transform(test_preds)
    test_preds_decoded_inversed = le.inverse_transform(test_preds_decoded)

    y_valid = le.inverse_transform(oe.inverse_transform(y_valid))

    '''
    i=0
    result = False
    print("\n\n-----------------------------------------------------")
    for x in y_valid:
        if(x==test_preds_decoded_inversed[i]):
            result = True
        else:
            result = False
        print("Initial: ", x, "==> Predicted: ", test_preds_decoded_inversed[i], " (", result,")")
        i+=1
    '''

    print("\n\nPerformance Metrics: \n-----------------------------------------")
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy: ", metrics.accuracy_score(y_valid, test_preds_decoded_inversed))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision: ", metrics.precision_score(y_valid, test_preds_decoded_inversed))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall: ", metrics.recall_score(y_valid, test_preds_decoded_inversed))

    # Model F1 Score: F1 Score might be a better measure to use 
    # if we need to seek a balance between Precision and Recall
    print("F1 Score: ", metrics.f1_score(y_valid, test_preds_decoded_inversed))

    # Model Specificity: a model's ability to predict true negatives of each available category
    print("Specificity: ", metrics.recall_score(y_valid, test_preds_decoded_inversed, pos_label=0))

    # Model Negative Predictive Value (NPV): 
    print("Negative Predictive Value (NPV): ", metrics.precision_score(y_valid, test_preds_decoded_inversed, pos_label=0))
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
    

    #plot the training and validation figures
    #accuracy = fitted_model.history['acc']
    #val_accuracy = fitted_model.history['val_accuracy']
    loss = fitted_model.history['loss']
    val_loss = fitted_model.history['val_loss']

    #plot the loss figure
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()