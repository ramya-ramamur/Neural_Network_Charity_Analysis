# Neural_Network_Charity_Analysis

# Analysis Overview
The purpose of this project is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by a charitable organization. Using the features in the provided dataset containing various measures on 34,000 organizations, we will analyze and classify the success of charitable donations with deep-learning neural networks with the TensorFlow platform in Python. 

We use the following methods for the analysis:
1. Preprocessing Data for a Neural Network Model
2. Compile, Train, and Evaluate the Model
3. Optimize the Model

# Resources
* Data Source: [charity_data.csv](https://github.com/ramya-ramamur/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv)
* Software: Python 3.8.8, Pandas Dataframe, Jupyter Notebook 6.4.6, Anaconda Navigator 2.1.1

# Results

### Data Preprocessing
The following preprocessing steps have been performed:
* The EIN and NAME columns have been dropped 
* The columns with more than 10 unique values have been grouped together 
* The categorical variables have been encoded using one-hot encoding 
* The preprocessed data is split into 
  - features : APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT are the features for our model.
  - target arrays : The column IS_SUCCESSFUL contains binary data is the target for our deep learning neural network.
* The preprocessed data is split into training and testing datasets 
* The numerical values have been standardized using StandardScaler() 

### Compile, Train, and Evaluate the Model
The neural network model using Tensorflow Keras contains working code that performs the following steps:
* The [neural network model] has 2 hidden layers. My first layer had 80 neurons, the second has 30 there is also an output layer. The first and second hidden layer have the "relu" activation function and the activation function for the output layer is "sigmoid."

The number of layers, the number of neurons per layer, and activation function are defined (2.5 pt)
An output layer with an activation function is created (2.5 pt)
There is an output for the structure of the model (5 pt)
There is an output of the model’s loss and accuracy (5 pt)
The model's weights are saved every 5 epochs (2.5 pt)
The results are saved to an HDF5 file (2.5 pt)
