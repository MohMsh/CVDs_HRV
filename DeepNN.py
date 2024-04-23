# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import class_weight
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras import regularizers

#===================================================================================================
#Optimal Parameters and results
#parameters
#Model2	SMOTE/STANDARD SCALING/test size:0.21/Layers:5/512,256,128,64,1/dropout:last layer 0.2/
#optimizer:SGD/learning rate:0.005/epochs: 10000/batch: 250/momentum: FALSE/class_weight: FALSE

#results
#accuracy: 90.19%/precision: 85.18%/Recall: 95.83%/F1 Score: 90.19%/Specificity: 85.18%/NPV: 95.83%
#===================================================================================================

def ApplyDeepNN(DataFrame):

        print("---------------------------------\n\nImplementing DeepNN on SHAREEDB\n\n---------------------------------")

        # Prepare X & Y Arrays
        X = DataFrame.iloc[:, 2:].values
        Y = DataFrame.iloc[:, 1].values

        # Transform the dataset using SMOTE
        oversample = SMOTE()
        X, Y = oversample.fit_resample(X, Y)

        # Split data into train and test arrays
        X_Train,test_X,Y_Train,test_Y = train_test_split(X, Y, test_size=0.21, random_state=42)
        
        # Feature Scaling
        sc_X = StandardScaler()
        X_Train = sc_X.fit_transform(X_Train)
        test_X = sc_X.transform(test_X)


        # Define the keras model
        MyModel = Sequential()
        MyModel.add(Dense(512, input_dim=26, activation='tanh'))
        #MyModel.add(Dropout(0.3))
        MyModel.add(Dense(256, activation='tanh'))
        #MyModel.add(Dropout(0.3))
        MyModel.add(Dense(128, activation='tanh'))
        #MyModel.add(Dropout(0.3))
        MyModel.add(Dense(64, activation='tanh'))
        MyModel.add(Dropout(0.18))
        MyModel.add(Dense(1, activation='sigmoid'))
        
        # Based on GridSearchCV optimization
        # The best learnrate and momentum are
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)#, momentum=0.1)
        #optimizer = tf.keras.optimizers.Adam(lr=0.005)

        # compile the keras model
        MyModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #using balanced weight negatively affected the performance of DNN
        
        #Calculate the class weight to balance the dataset
        #Keep the format of the following 4 lines as they are
        #for a reason related to t Keras version, usual format didn't work
        #check this post: https://stackoverflow.com/questions/69783897/compute-class-weight-function-issue-in-sklearn-library-when-used-in-keras-cl
        class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(Y_Train),
                                        y = Y_Train                                                    
                                    )
        class_weights = dict(zip(np.unique(Y_Train), class_weights))     
        
        # fit the keras model on the dataset
        fitted_model = MyModel.fit(X_Train, Y_Train, epochs=6850, batch_size=250,
                                   verbose=1, validation_data=(test_X, test_Y),
                                   shuffle=True)#,class_weight=class_weights)

        # make class predictions with the model
        predictions = (MyModel.predict(test_X) > 0.5).astype(int)
        
        #Evaluate the model to get the accuracy and loss
        test_eval = MyModel.evaluate(test_X, test_Y, verbose=1)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
            
        #plot the accuracy and loss plots between training and validation data
        accuracy = fitted_model.history['accuracy']
        val_accuracy = fitted_model.history['val_accuracy']
        loss = fitted_model.history['loss']
        val_loss = fitted_model.history['val_loss']

        # Making the Confusion Matrix 
        cm = confusion_matrix(test_Y, predictions)
        print("Confusion Matrix",cm)
        #calculate TP, TN, FP and FN
        TP = cm[0,0]
        TN = cm[1,1]
        FP = cm[0,1]
        FN = cm[1,0]

        print("True Positives: ", TP)
        print("True Negatives: ", TN)
        print("False Positives: ", FP)
        print("False Negatives: ", FN)
        

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy: ", metrics.accuracy_score(test_Y, predictions))
        
        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision: ", metrics.precision_score(test_Y, predictions))
        
        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall: ", metrics.recall_score(test_Y, predictions))
        
        # Model F1 Score: F1 Score might be a better measure to use 
        # if we need to seek a balance between Precision and Recall
        print("F1 Score: ", metrics.f1_score(test_Y, predictions))

        # Model Specificity: a model's ability to predict true negatives of each available category
        print("Specificity: ", metrics.recall_score(test_Y, predictions, pos_label=0))
        
        # Model Negative Predictive Value (NPV): 
        print("Negative Predictive Value (NPV): ", metrics.precision_score(test_Y, predictions, pos_label=0))        
        
        # print classification error
        #classification_error = (FP + FN) / float(TP + TN + FP + FN)
        #print('Classification error : {0:0.4f}'.format(classification_error))
        print("===============================================================================\n")
        print(classification_report(test_Y, predictions))
        print("===============================================================================\n")
        #==========================================================================================
        
        
        #Ask wether to save the model or not
        SaveModel = input("---------------------------------\n\nDo you want to save the model? (Y/N):")
        if(SaveModel == "Y".upper()):
            #==========================================================================================
            #save the model to be used later
            MyModel.save("SavedModels/DeepNNModel.h5py")
            #==========================================================================================
        
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        '''
        # print the predictions
        for i in range(len(predictions)):
                print('%s => %d (expected %d)\n' % (str(i), predictions[i], test_Y[i]))
        '''