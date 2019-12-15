#!/usr/bin/env python
# coding: utf-8

# # Diabetes Evaluation
# The last step in our project is being able to take our diabetes patients and assess several models upon them to test the accuracy of being able to classify a patient with diabetes compared to one who does not using the metrics that we have from social determinants and their patient vitals. There are two types of models that will be tested but 3 overall. The first type of models are classification models; K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). For each of these models we will compare the results to one another with the goal being able to know distinctly is there a difference that the algorithms can denote between the two different types of patients. The second type of model will be a Neural Network, which will be used to help predict the potential onset of diabetes with our patients based on the metrics that have been collected.

# ## KNN and SVC Models
# I will start with the KNN and SVC models and then assess and compare each in their ability to classify which patients have diabetes and which ones do not.

# In[16]:


#Import libraries
import pandas as pd
import os


# In[42]:


#Bring in the data
diabetes = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Diabetes1.csv")
diabetes.head()


# In[43]:


#Import the necessary packages to create the classification KNN & SVC models
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

#We can print the type of version with the code below, this is just useful info to have incase we need to update something.
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# In[44]:


#Now, let's bring in different aspects of the packages which will be used specifically for our models.
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
#Don't worry about any warnings that may possible come in, as long as you don't have any errors that is what is important.


# In[45]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[46]:


df= diabetes
df.describe()
#It apppears we have some data points that are null. We will need to go through a preprocessing stage to make sure we have no null values.


# In[47]:


#Preprocessing the data is what we will do in this cell, if there is any missing data we need to take care of it or 
# it will get in the way of our model.
df.dropna(inplace=True)
print(df.axes)

df.drop(['PAT_ID'], 1, inplace=True)
#Replacing and dropping values are very important in the preprocessing of data before modeling, it can be a tedious
# task but if done right will lead to the best outputs.

#Print the shape of the database (dimensions)
print(df.shape)
#Showing the shape will show that we have 24 columns. 23 metrics and 1 binary class which will state if the patient does
# or doesn't have our chronic condition (diabetes)


# In[48]:


#Before we model the data, let's take some time to explore it and see what each variable has to offer.
print(df.loc[5])
#Will visualize the our 5th row of data.

#Print the shape of the dataset
print(df.shape)
#We will see in our output is the dimensions of our data. The line below this, will give us a summary of each measure 
# within our dataset
print(df.describe())
#You'll be able to see the count of each of our variables, which should all be the same after our preprocessing step.
#Also, you will be able to see the mean, standard deviation, minimum, 25%, 50%, & 75% limit and max values for each
#measure.


# In[49]:


#Plot histograms showing the distribution of each variable
df.hist(figsize = (16,16))
plt.show()
#From the looks of our distributions, there doesn't appear to be many that would be termed as normal, many are skewed 
#either postively or negatively.


# In[50]:


#A scatter matrix is also great to show correlation between different variables.
scatter_matrix(df, figsize = (18, 18))
plt.show()


# In[51]:


#Let's now create an X and Y datasets for training 
#Before we need to drop our Diabetes Codes and keep only that for our Y labels.
X = np.array(df.drop(['Diabetes Codes'], 1))
Y = np.array(df['Diabetes Codes'])

#Here I will make it an 80:20 Training/Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[52]:


#Testing Options will now be set and we will be scoring for accuracy
seed = 10
scoring = 'accuracy'


# In[53]:


#Define models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC(gamma = 'scale')))

#Evaluate each model in order
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Looking at our accuracy scores for both models it appears the results are quite promising. The KNN model did terrific with a score of 90.6% accuracy and a standard deviation of 0.03. While the SVM was at 84.8% and the standard deviation was at 0.04. So both results not bad for the training portion of our analysis.

# In[54]:


#Let's make predictions on the validation dataset now to see how close we are to our training set.

for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    
#Some terms for any who need them:
#Accuracy - ratio of correctly predicted observations to the total observations.
#Precision - (false positives) ratio of correctly predicted positive observations to the total predicted positive observations
#Recall - (false negatives) ratio of correctly predicted postive observations to the all observations in actual class - yes.
#F1 Score - F1 Score is the weighted average of Precision and Recall. This score takes both false postives and false negatives into account.


# ## Evaluating the Classifier Models
# Examining our results we have some encouraging points and some where some reflection will need to be done. For the KNN model the results were fantastic. With an overall accuracy of about 90% and a precision that is 90% for non-diabetic patients and 80% for diabetic patients, while the recall was outstanding at 99% for non-diabetic patients but lackluster for diabetic patients at 0.33. Then the F1 score was again great for non-diabetic patients but not so much for diabetic patients. The results of the SVM model were promising when it comes to overall accuracy of 86% but the precision, recall and F1 Score didn't populate for patients with diabetes which means in the testing set it had no patients that were diabetic so the distribution between the training and testing set wasn't even with the data set that we have. We could fix this issue by cutting down our number of samples so that the distribution is better but with the performance of the KNN and the accuracy score of the SVM I think it is encouraging the results which were shown in the first run. 

# ## Neural Network
# Putting together a Neural Network to try to predict obesity patients could be yield some interesting results. The results of something like this could be monumental in helping patients become more proactive in their healthcare outcomes and give them and their providers and opportunity to correct an issue before it becomes truly debiliating for the patient and their lifestyles.

# In[55]:


#Upload the packages needed for our Neural Network
import sys
import pandas
import numpy
import sklearn
import keras

print('Python: {}'.format(sys.version))
print('Pandas: {}'.format(pandas.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
print('Keras: {}'.format(keras.__version__))
#I know a lot of these packages are already loaded in from the work done on he KNN and SVC but I like to be reassured about these sorts of things.


# In[56]:


#Let's also bring back in the data here, that way we can start with a clean version of the data and manipulate it properly 
# for our Neural Net and not have any issues potentially using the same data from the KNN an SVC models.
import pandas as pd
import numpy as np

#Bring in the data
ds = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Diabetes1.csv")
ds.head()


# In[57]:


#From our classifiers we know we have some missing and null data, so let's drop that and then check those are gone.
ds.drop(['PAT_ID'], 1, inplace=True)
ds.dropna(inplace = True)

#Summarize our data
ds.describe()
#Our results show we have 428 patients within our dataset for all fields.


# In[58]:


#Convert dataframe to a numpy array
data = ds.values
print(data.shape)


# In[60]:


#Split our data into inputs of X and outputs of Y
X = data[:, 0:22]
Y = data[:, 23].astype(int)

#Print the shapes of each now, and an example row to show what we have for our outputs Y
print(X.shape)
print(Y.shape)
print(Y[:5])


# In[61]:


#It is better for our data to be put through a normalization process. 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)


# In[62]:


print(scaler)


# In[63]:


#Transform and show the training dataset
X_standardized = scaler.transform(X)

nn_diab = pd.DataFrame(X_standardized)
nn_diab.describe()


# In[64]:


#Its now time to import the necessary packages from sklear and keras to perform our Neural Net
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[66]:


#Start definining or Neural Network
def create_model():
    #Create the model
    model = Sequential()
    model.add(Dense(20, input_dim = 20, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(10, input_dim = 10, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model 
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())
#It looks like the parameters of our Neural net are going to be extensive with over 641 trainable params.


# In[68]:


#Do a grid search for the optimal batch size and number of epochs 

#Define a random seed
seed = 6
np.random.seed(seed)

#Start definining the model
def create_model():
    #Create model
    model = Sequential()
    model.add(Dense(22, input_dim = 22, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model 
model = KerasClassifier(build_fn = create_model, verbose = 1)

#Define the grid search parameters
batch_size = [20, 40, 80]
epochs = [10, 50, 100]

#Make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

#Build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

#Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# The accuracy of our model is in the training set is in the high 90% range and our testing data only drops to perform best at 91.6% when our batch_size is at 80 and our epochs is at 50 so with a test range which is still really impressive for our first go around, let's now try and find what the dropout rate is for our model and see what kind of information we can pull from that.

# In[69]:


#Perform a grid search again for the learning rate and dropout rate.
#Import the Dropout package from keras
from keras.layers import Dropout

#Define the random seed
seed = 6
np.random.seed(seed)

#Start defining the model
def create_model(learn_rate, dropout_rate):
    #create model
    model = Sequential()
    model.add(Dense(22, input_dim = 22, kernel_initializer = 'normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation= 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 50, batch_size = 80, verbose = 0)

#Define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2]

#Make a dictionary of the grid search parameters
param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate)

#Build and fit th Grid search
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

#Summarize the results 
print("Best:{0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# The best results come in again around 91.8% and it is with a learn rate of 0.001 and a dropout rate of 0.1. 
# 
# Next it will be time to find the optimal size for initilization and activation metrics.

# In[70]:


#Do a grid search to optimize the kernel initilization and activation functions
seed = 6
np.random.seed(seed)

#Start defining the model
def create_model(activation, init):
    #create model
    model = Sequential()
    model.add(Dense(22, input_dim = 22, kernel_initializer = 'normal', activation='relu'))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 50, batch_size = 80, verbose = 0)

#Define the grid search parameters
activation = ['softmax', 'relu','tanh','linear']
init = ['uniform','normal','zero']

#Make a dictionary of the grid search parameters
param_grid = dict(activation = activation, init = init)

#Build & fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

#Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# This go around the best results were at 92.5% and it was when the activation was at softmax and the initializer was at zero. Now combined with what values we have gained prior we are ready to set up the neuron values.

# In[72]:


#Let's now fine tune the model and find the optimal number of neurons in each hidden layer

#Define a random seed
seed = 6
np.random.seed(seed)

#Define the model 
def create_model(neuron1, neuron2):
    #Create the model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 22, kernel_initializer = 'zero', activation = 'softmax'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer = 'zero', activation = 'softmax'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 50, batch_size = 80, verbose = 0)

#Define the grid search parameters
neuron1 = [4, 8, 16]
neuron2 = [2, 4, 8]

#Make a dictionary of the grid search parameters 
param_grid = dict(neuron1 = neuron1, neuron2 =  neuron2)

#Build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), refit = True, verbose =10)
grid_results = grid.fit(X_standardized, Y)

#Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# Now finally, the best result was when neuron1 = 4 and neuron2 = 4 as well with a percentage of 84.3% accuracy. The final step is to now predict the liklihood of patients developing diabetes.

# In[73]:


#Generate predictions with the optimal parameters
y_pred = grid.predict(X_standardized)


# In[74]:


print(y_pred.shape)


# In[75]:


print(y_pred[:5])


# In[76]:


#Generate a classification report
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))


# The accuracy of the model did good overall, predicting at 0.84 is pretty high for the first time around. It is troubling to see the precision, recall and F1 score are each at 0.00 but that could be because of the distribution of non-diabetic patients being so much more than diabetic patients. When going through again I'll have to re-assess different aspects of my models and work to correct that first before diving into the individual variable performance. 

# In[ ]:




