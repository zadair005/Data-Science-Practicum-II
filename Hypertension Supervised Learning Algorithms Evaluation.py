#!/usr/bin/env python
# coding: utf-8

# # Hypertension Evaluation
# Our final steps in our project are to take our hypertension patients and assess several models upon them to test the accuracy of how well the metrics we've chosen will predict if a patient has that particular disease or not. We will start by looking at KNN and SVC models and the accuracy and precision of the model will be very important here. Finally we will create a neural network on the data which will help us predict the chronic condition being in our patients. For this portion of the examination we made a couple of alterations to our data. Instead of looking at the three chronic conditions at one time over the last 3 years I took a sample of the data for the last year and then I set a classifier field which denotes if a patient does have hypertension (1) or does not (0).

# ## KNN and SVC Models
# We will construct the KNN and SVC models and then assess the similarities and differences in their ability to classify if which patients have hypertension and which ones do not.

# In[137]:


#Import libraries
import pandas as pd
import os


# In[138]:


#Bring in the data
hypertension = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Hypertension1.csv")
hypertension.head()


# In[139]:


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


# In[140]:


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


# In[141]:


df= hypertension
df.describe()
#It apppears we have some data points that are null. We will need to go through a preprocessing stage to make sure we have no null values.


# In[142]:


#Preprocessing the data is what we will do in this cell, if there is any missing data we need to take care of it or 
# it will get in the way of our model.
df.dropna(inplace=True)
print(df.axes)

df.drop(['PAT_ID'], 1, inplace=True)
#Replacing and dropping values are very important in the preprocessing of data before modeling, it can be a tedious
# task but if done right will lead to the best outputs.

#Print the shape of the database (dimensions)
print(df.shape)
#Showing the shape will show that we have 25 columns. 24 metrics and 1 binary class which will state if the patient does
# or doesn't have our chronic condition (hypertension)


# In[143]:


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


# In[144]:


#Plot histograms showing the distribution of each variable
df.hist(figsize = (16,16))
plt.show()
#From the looks of our distributions, there doesn't appear to be many that would be termed as normal, many are skewed 
#either postively or negatively.


# In[145]:


#A scatter matrix is also great to show correlation between different variables.
scatter_matrix(df, figsize = (18, 18))
plt.show()


# In[148]:


#Let's now create an X and Y datasets for training 
#Before we need to drop our Hypertension Codes and then only have it for our labels.
X = np.array(df.drop(['Hypertension Codes'], 1))
Y = np.array(df['Hypertension Codes'])

#Here I will make it an 80:20 Training/Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[149]:


#Testing Options will now be set and we will be scoring for accuracy
seed = 10
scoring = 'accuracy'


# In[152]:


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


# Looking at the accuracy scores of both models is highly encouraging. It appears our KNN model has an accuracy score of about 90.9% with a standard deviation of about 0.04 and our Support Vector Machine has an accuracy score of 84.0% and a standard deviation of 0.06 as well. Let's now look and see this more detailed but these are great results for our initial round of modeling.

# In[153]:


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


# Diving deeper there is more information to pull from the KNN and SVM models. The KNN model had very high precision numbers when predicting when someone doesn't have Hypertension, and decent when numbers when predicting that someone does have hypertension. The SVM model doesn't do as well overall but only because the recall and F1 score of the model is very low when predicting a patient does have hypertension.

# With the SVC model perfoming lower than the KNN lets see if we can improve that a little bit here.

# In[154]:


clf = SVC(gamma = 'scale') 

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1,3,2,4,5,1,3,3,5,8,7,3,1,1,2,5]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)


# Unfortunately the reshape of our model yielded no better results, but the results to start were encouraging for starting out.

# ## Evaluating the Classifier Models
# The performance of the models has some encouraging signs for it's first go around. We saw the KNN model did pretty good with an accuracy at 93%. Anytime you can get over 80% you should be very happy to see that. Then the precision recall and F1 scores were all pretty good as well each scoring in a decently high end, some adjustments coul be made to improve the model overall but it was exciting to see. The SVM performed well but not as good as the KNN model with an accuracy of 88.2% and some troubling recall, and F1 scores. But I believe if we assess our variables, maybe remove some low contributors and look to find some better ones that might fit more for hypertension patients we could see better results. But for a first round this I believe is very encouraging. Now, let's move onto creating a neural network and try to predict the onset of hypertension in patients.

# ## Neural Network
# Putting together a Neural Network to try to predict hypetension patients could be yield some interesting results. The results of something like this could be monumental in helping patients become more proactive in their healthcare outcomes and give them and their providers and opportunity to correct an issue before it becomes truly debiliating for the patient and their lifestyles.

# In[155]:


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


# In[156]:


#Let's also bring back in the data here, that way we can start with a clean version of the data and manipulate it properly 
# for our Neural Net and not have any issues potentially using the same data from the KNN an SVC models.
import pandas as pd
import numpy as np

#Bring in the data
ht = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Hypertension1.csv")
ht.head()


# In[157]:


#From our classifiers we know we have some missing and null data, so let's drop that and then check those are gone.
ht.drop(['PAT_ID'], 1, inplace=True)
ht.dropna(inplace = True)

#Summarize our data
ht.describe()
#Our results show we have 508 patients within our dataset for all fields.


# In[158]:


#Convert dataframe to a numpy array
data = ht.values
print(data.shape)


# In[159]:


#Split our data into inputs of X and outputs of Y
X = data[:, 0:23]
Y = data[:, 24].astype(int)

#Print the shapes of each now, and an example row to show what we have for our outputs Y
print(X.shape)
print(Y.shape)
print(Y[:5])


# In[160]:


#It is better for our data to be put through a normalization process. 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)


# In[161]:


print(scaler)


# In[162]:


#Transform and show the training dataset
X_standardized = scaler.transform(X)

nn_hyper = pd.DataFrame(X_standardized)
nn_hyper.describe()


# In[163]:


#Its now time to import the necessary packages from sklear and keras to perform our Neural Net
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[164]:


#Start definining or Neural Network
def create_model():
    #Create the model
    model = Sequential()
    model.add(Dense(23, input_dim = 23, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model 
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())
#It looks like the parameters of our Neural net are going to be extensive with over 800 trainable params.


# In[165]:


#Do a grid search for the optimal batch size and number of epochs 
​#Do a grid search for the optimal batch size and number of epochs 

#Define a random seed
seed = 6
np.random.seed(seed)

#Start definining the model
def create_model():
    #Create model
    model = Sequential()
    model.add(Dense(23, input_dim = 23, kernel_initializer = 'normal', activation = 'relu'))
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
#Define a random seed
seed = 6
np.random.seed(seed)

#Start definining the model
def create_model():
    #Create model
    model = Sequential()
    model.add(Dense(23, input_dim = 23, kernel_initializer = 'normal', activation = 'relu'))
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


# The accuracy of our model is in the training set is in the 90% range and our testing data only drops to perform best at 81.7% when our batch_size is at 80 and our epochs is at 10 so with a test range which is still really impressive for our first go around, let's now try and find what the dropout rate is for our model and see what kind of information we can pull from that.

# In[166]:


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
    model.add(Dense(23, input_dim = 23, kernel_initializer = 'normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation= 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 80, verbose = 0)

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


# We see that our best results appear when we have a dropout rate of 0.1 and a learn rate of 0.001 with the result being at 81.1% accuracy which again is really good for this first go around. 
# 
# Our next step will be to optimize the weight initialization that we're applying to the end of each of the neurons.

# In[167]:


#Do a grid search to optimize the kernel initialization and activation functions
seed = 6
np.random.seed(seed)

#Start defining the model
def create_model(activation, init):
    #create model
    model = Sequential()
    model.add(Dense(23, input_dim = 23, kernel_initializer = 'normal', activation='relu'))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

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


# The best results we received were 81.3% and it was when we had our activation set to linear and our initializer set to uniform. Now that we have our best learn rate, dropout rate, activation and initializer we can use all of this to help fully optimize the result of our neural network.

# In[168]:


#Let's now fine tune the model and find the optimal number of neurons in each hidden layer

#Define a random seed
seed = 6
np.random.seed(seed)

#Define the model 
def create_model(neuron1, neuron2):
    #Create the model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 23, kernel_initializer = 'uniform', activation = 'linear'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer = 'uniform', activation = 'linear'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

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


# Running this through again we found that our best result of 81.9% and it came when our neuron1 was set to 4 and neuron2 was set to 2. Our final step is to predict the probability of patients developing hypertension.

# In[169]:


#Generate predictions with the optimal hyperparameters
y_pred = grid.predict(X_standardized)


# In[170]:


print(y_pred.shape)


# In[171]:


print(y_pred[:5])


# In[172]:


#Generate a classification report
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))


# The results of our model are quite good overall, we see that from the dataset we have, that we have an overall accuracy of 0.93 to predict the liklihood of someone developing hypertension with our precision for someone to not have it at 95% and the precision of them not to have it at 77%. There are ways to improve on these numbers but the first round of results is very encouraging. I believe if we wanted to try to improve this models performance we would need to do something similar to what I suggested needs to be done with the classifier models. Refine the social determinants some and find more patient variables picked up in the hospital which could give us even better results than we have here.

# In[ ]:




