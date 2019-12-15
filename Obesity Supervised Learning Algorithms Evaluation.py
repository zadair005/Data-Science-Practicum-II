#!/usr/bin/env python
# coding: utf-8

# # Obesity Evaluation
# Continuing with the final step in this project it is time to use several supervised learning algorithms to try to accurately classify patients with obesity from those who don't have the chronic condition, and then attempt to predict the probability of someone developing this disease if they do not have it based on the information that has already been collected. To start, the classification models used here will be the K Nearest Neighbors (KNN) and the Support Vector Machine (SVM) which will assess an accuracy based on their performance than measure its precision, recall and F1 score to give a more accurate measurement of performance. To end it, a Neural Network will be used to attempt to accurately predict the liklihood of someone developing obesity as a chronic condition. The accuracy and precision of this model will be important to see if this can be something that can be rolled out and used to help doctors better examine patients as they come in and reaffirm intuition or help them see where the patient might be in real danger. 

# ## KNN and SVC Models
# We will construct the KNN and SVC models and then assess the similarities and differences in their ability to classify if a patient has obesity or not.

# In[1]:


#Import libraries
import pandas as pd
import os


# In[9]:


#Bring in the data
obesity = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Obesity1.csv")
obesity.head()


# In[10]:


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


# In[11]:


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


# In[12]:


df= obesity
df.describe()
#It apppears we have some data points that are null. We will need to go through a preprocessing stage to make sure we have no null values.


# In[13]:


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
# or doesn't have our chronic condition (hypertension)


# In[14]:


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


# In[15]:


#Plot histograms showing the distribution of each variable
df.hist(figsize = (16,16))
plt.show()
#From the looks of our distributions, there doesn't appear to be many that would be termed as normal, many are skewed 
#either postively or negatively.


# In[16]:


#A scatter matrix is also great to show correlation between different variables.
scatter_matrix(df, figsize = (18, 18))
plt.show()


# In[17]:


#Let's now create an X and Y datasets for training 
#Before we need to drop our Hypertension Codes.
X = np.array(df.drop(['Obesity Codes'], 1))
Y = np.array(df['Obesity Codes'])

#Here I will make it an 80:20 Training/Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[18]:


#Testing Options will now be set and we will be scoring for accuracy
seed = 10
scoring = 'accuracy'


# In[23]:


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


# The Accuracy scores for our two models are quite different. The KNN model has a terrific accuracy score of 90% and a standard deviation of 0.063 while the SVM didn't perform as well, only posting an accuracy score of 71.3% and a standard deviation of 0.06. Next is where the predictions will be made on the validation training set to see how well these models perform in that regard.

# In[24]:


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


# Looking deeper into our models are trends noted from earlier continue. Our KNN again performs well having a great score with high precision, recall, and f1 scores but because of a warning in Jupyter notebook that I thought I corrected I'm unable to see the precision,recall and F1 scores for my SVM, but with the SVM having an accuracy of .73 I can only assume that those are fairly low.

# ## Evaluating our Classifier Models
# The performance of the KNN model was great, an accuracy score over 90% and strong supporting metrics I have a lot of confidence in that model moving forward. Unfortunately the SVM model really struggled with an overall accuracy of just 73% and some decent numbers when measuring the precision, recall and F1 score but it is really unfortunate the UndefinedMetricWarning has prevented seeing the results of the predicted values. Moving forward now, it's tim to use a neural network to try and predict the onset of obesity in patients.

# ## Neural Network
# Putting together a Neural Network to try to predict obesity patients could be yield some interesting results. The results of something like this could be monumental in helping patients become more proactive in their healthcare outcomes and give them and their providers and opportunity to correct an issue before it becomes truly debiliating for the patient and their lifestyles.

# In[87]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
#I know a lot of these packages are already loaded in from the work done on he KNN and SVC but I like to be reassured about these sorts of things


# In[88]:


#Let's also bring back in the data here, that way we can start with a clean version of the data and manipulate it properly 
# for our Neural Net and not have any issues potentially using the same data from the KNN an SVC models.
import pandas as pd
import numpy as np

#Bring in the data
ob = pd.read_csv("I:\Projects\BrunerClinic\ZA Practicum\Alteryx Workflows\Alteryx Outputs\Files for Supervised Algorithms\Data For KNN and SVC Models\BrunerClinic_Obesity1.csv")
ob.head()


# In[89]:


#From our classifiers we know we have some missing and null data, so let's drop that and then check those are gone.
ob.drop(['PAT_ID'], 1, inplace=True)
ob.dropna(inplace = True)

#Summarize our data
ob.describe()
#Our results show we have 502 patients within our dataset for all fields.


# In[90]:


#Convert dataframe to a numpy array
data = ob.values
print(data.shape)


# In[91]:


#Split our data into inputs of X and outputs of Y
X = data[:, 0:22]
Y = data[:, 23].astype(int)

#Print the shapes of each now, and an example row to show what we have for our outputs Y
print(X.shape)
print(Y.shape)
print(Y[:5])


# In[92]:


#It is better for our data to be put through a normalization process. 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
print(scaler)


# In[93]:


#Transform and show the training dataset
X_standardized = scaler.transform(X)

nn_ob = pd.DataFrame(X_standardized)
nn_ob.describe()


# In[94]:


#Its now time to import the necessary packages from sklear and keras to perform our Neural Net
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[ ]:





# In[95]:


#Start definining or Neural Network
def create_model():
    #Create the model
    model = Sequential()
    model.add(Dense(22, input_dim = 22, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(11, input_dim = 11, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile the model 
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())
#It looks like the parameters of our Neural net are going to be extensive with 771 trainable params.


# In[96]:


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


# The accuracy of our model within the training set was in the high 90% range and in our testing the results only dropped slightly. Our best performance within our testing set came when we have a batch size of 80 and epochs = 50. Next the goal will be to find the ideal dropout rate and learning rate for this neural network.

# In[97]:


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


# Again were seeing terrific results, with the neural network performing best when the dropout rate is set to 0.0 and the learn rate is at 0.1 the neural network's accuracy is measured at 95.8%. Now its time to find the optimal kernel initialization and activation function for this neural net.

# In[98]:


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
    adam = Adam(lr = 0.1)
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
    print('{0} ({1}) with {2}'.format(mean, stdev, param))


# The best results again were at 95.8% with an activation of softmax and an initializer of zero so these can be added to our Neural net and can now tune the neurons.

# In[76]:


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
    adam = Adam(lr = 0.1)
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


# Interesting to look at the results here of the neural network, considering our model had bee over 95% but has now dropped to 71.9% and it doesn't matter the value of the neurons.

# In[99]:


#Generate predictions with the optimal hyperparameters
y_pred = grid.predict(X_standardized)

print(y_pred.shape)


# In[100]:


print(y_pred[:5])


# In[101]:


#Generate a classification report
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))


# The results of our neural network for the obesity patients looks highly promising but I suspect an issue with our result. The fact that it has such a high accuracy along with nearly perfect precision, recall and f1 scores leads me to believe there is an issue with these model overall and may need to be examined deeper to see where corrections can be made to provide an accurate assessment of model performance before moving forward to seeing what metrics are better to uses and not use when trying to predict obesity of a patient. 

# In[ ]:




