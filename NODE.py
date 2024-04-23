import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from typing import Union, Optional
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from sklearn import metrics
from numpy import mean, std
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import joblib
import warnings

#Turn off all warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

@tf.function
def sparsemoid(inputs: tf.Tensor):
    return tf.clip_by_value(0.5 * inputs + 0.5, 0., 1.)

@tf.function
def identity(x: tf.Tensor):
    return x

class ODST(tf.keras.layers.Layer):
    def __init__(self, n_trees: int = 3, depth: int = 4, units: int = 1, threshold_init_beta: float = 1.):
        super(ODST, self).__init__()
        self.initialized = False
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
    
    def build(self, input_shape: tf.TensorShape):
        feature_selection_logits_init = tf.zeros_initializer()
        self.feature_selection_logits = tf.Variable(initial_value=feature_selection_logits_init(shape=(input_shape[-1], self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)        
        
        feature_thresholds_init = tf.zeros_initializer()
        self.feature_thresholds = tf.Variable(initial_value=feature_thresholds_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        log_temperatures_init = tf.ones_initializer()
        self.log_temperatures = tf.Variable(initial_value=log_temperatures_init(shape=(self.n_trees, self.depth), dtype='float32'),
                                 trainable=True)
        
        indices = tf.keras.backend.arange(0, 2 ** self.depth, 1)
        offsets = 2 ** tf.keras.backend.arange(0, self.depth, 1)
        bin_codes = (tf.reshape(indices, (1, -1)) // tf.reshape(offsets, (-1, 1)) % 2)
        bin_codes_1hot = tf.stack([bin_codes, 1 - bin_codes], axis=-1)
        self.bin_codes_1hot = tf.Variable(initial_value=tf.cast(bin_codes_1hot, 'float32'),
                                 trainable=False)
        
        response_init = tf.ones_initializer()
        self.response = tf.Variable(initial_value=response_init(shape=(self.n_trees, self.units, 2**self.depth), dtype='float32'),
                                 trainable=True)
                
    def initialize(self, inputs):        
        feature_values = self.feature_values(inputs)
        
        # intialize feature_thresholds
        percentiles_q = (100 * tfp.distributions.Beta(self.threshold_init_beta, 
                                                      self.threshold_init_beta)
                         .sample([self.n_trees * self.depth]))
        flattened_feature_values = tf.map_fn(tf.keras.backend.flatten, feature_values)
        init_feature_thresholds = tf.linalg.diag_part(tfp.stats.percentile(flattened_feature_values, percentiles_q, axis=0))
        
        self.feature_thresholds.assign(tf.reshape(init_feature_thresholds, self.feature_thresholds.shape))
        
        
        # intialize log_temperatures
        self.log_temperatures.assign(tfp.stats.percentile(tf.math.abs(feature_values - self.feature_thresholds), 50, axis=0))
        
        
        
    def feature_values(self, inputs: tf.Tensor, training: bool = None):
        feature_selectors = tfa.activations.sparsemax(self.feature_selection_logits)
        # ^--[in_features, n_trees, depth]

        feature_values = tf.einsum('bi,ind->bnd', inputs, feature_selectors)
        # ^--[batch_size, n_trees, depth]
        
        return feature_values
        
    def call(self, inputs: tf.Tensor, training: bool = None):
        if not self.initialized:
            self.initialize(inputs)
            self.initialized = True
            
        feature_values = self.feature_values(inputs)
        
        threshold_logits = (feature_values - self.feature_thresholds) * tf.math.exp(-self.log_temperatures)

        threshold_logits = tf.stack([-threshold_logits, threshold_logits], axis=-1)
        # ^--[batch_size, n_trees, depth, 2]

        bins = sparsemoid(threshold_logits)
        # ^--[batch_size, n_trees, depth, 2], approximately binary

        bin_matches = tf.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # ^--[batch_size, n_trees, depth, 2 ** depth]

        response_weights = tf.math.reduce_prod(bin_matches, axis=-2)
        # ^-- [batch_size, n_trees, 2 ** depth]

        response = tf.einsum('bnd,ncd->bnc', response_weights, self.response)
        # ^-- [batch_size, n_trees, units]
        
        return tf.reduce_sum(response, axis=1)

class NODE(tf.keras.Model):
    def __init__(self, units: int = 1, n_layers: int = 1, link: tf.function = tf.identity, n_trees: int = 3, depth: int = 4, threshold_init_beta: float = 1., feature_column: Optional[tf.keras.layers.DenseFeatures] = None):
        super(NODE, self).__init__()
        self.units = units
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.depth = depth
        self.units = units
        self.threshold_init_beta = threshold_init_beta
        self.feature_column = feature_column
        
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
        
        self.bn = tf.keras.layers.BatchNormalization()
        self.ensemble = [ODST(n_trees = n_trees,
                              depth = depth,
                              units = units,
                              threshold_init_beta = threshold_init_beta) 
                         for _ in range(n_layers)]
        
        self.link = link
        
        
    def call(self, inputs, training=None):
        X = self.feature(inputs)
        X = self.bn(X, training=training)
        
        for tree in self.ensemble:
            H = tree(X)
            X = tf.concat([X, H], axis=1)
            
        return self.link(H)


def ApplyNODE(DataFrame):

    print("---------------------------------\n\nImplementing NODE on SHAREEDB\n\n---------------------------------")

    le = LabelEncoder()
    oe = OneHotEncoder()
    CATEGORICAL_COLUMNS = []
    NUMERIC_COLUMNS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    # Splitting Data Between Features & Results
    X = DataFrame.iloc[:, 2:]
    Y = DataFrame.iloc[:, 1:2]

    # Transform the dataset using SMOTE
    oversample = SMOTE()
    X, Y = oversample.fit_resample(X, Y)

    numerical_features = X.columns

    # Splitting the dataset into the Training set and Test set
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y,stratify=Y, test_size = 39, random_state=42)

    #Data Transformation
    y_train = le.fit_transform(y_train)
    y_valid = le.fit_transform(y_valid)

    unique, counts = np.unique(y_train, return_counts=True)
    unique, counts = np.unique(y_valid, return_counts=True)

    y_train = oe.fit_transform(y_train.reshape(-1, 1))
    y_train = y_train.toarray()
    y_train = pd.DataFrame(y_train)

    y_valid = oe.fit_transform(y_valid.reshape(-1, 1))
    y_valid = y_valid.toarray()
    y_valid = pd.DataFrame(y_valid)

    RobustScaler_transformer = RobustScaler().fit(x_train.values)
    X_trainRobustScaler = RobustScaler_transformer.transform(x_train.values)
    X_validRobustScaler = RobustScaler_transformer.transform(x_valid.values)

    x_train =pd.DataFrame(
        X_trainRobustScaler,
        columns=list(numerical_features))

    x_valid =pd.DataFrame(
        X_validRobustScaler,
        columns=list(numerical_features))
        
    #Best parameters till now are:
    #layers 8
    #depth 10
    #n_trees 1
    #threshold_init_beta 1
    #learning_rate 5
    #validation_split 0.79
    #batch_size 26

    node = NODE(n_layers=10,#best 5
                units=2,
                depth=15, #was 10
                n_trees=1,
                threshold_init_beta = 1,
                link=tf.keras.activations.sigmoid)

    #Define the optimizer
    opt = keras.optimizers.SGD(learning_rate=5) #best 0.1

    node.compile(optimizer=opt,
                loss='bce')
                
    fitted_model = node.fit(x=x_train,
                            y=y_train,
                            validation_split=0.79,
                            shuffle=True,
                            batch_size=1040, #best 26
                            epochs=7500,
                            verbose=1)

    print(node.summary())
    print("\n\n")

    preds = node.predict(x_valid)
    test_preds_decoded = oe.inverse_transform(preds)
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