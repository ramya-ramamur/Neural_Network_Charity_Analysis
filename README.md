Use TensorFlow and deep learning neural networks to analyze and classify the success of charitable donations.

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
1. The [neural network model](https://github.com/ramya-ramamur/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity.ipynb) model using Tensorflow Keras contains working code that performs the following steps:
* The neural network has 2 hidden layers. The first layer has 80 neurons, the second has 30. 
* The input data has 43 features and 25,724 samples. Activation function used is ReLU for the hidden layers. 
* The output layer is a binary classification. The activation function for the output layer is Sigmoid.
* For the compilation, the optimizer is adam and the loss function is binary_crossentropy.

<img width="1028" alt="Screen Shot 2022-03-21 at 8 33 26 AM" src="https://user-images.githubusercontent.com/75961057/159295845-2ea0babf-e6a8-4ed9-93c6-9035044b4947.png">

2. The model callback saves the model's weights every 5 epochs.
<img width="1135" alt="Screen Shot 2022-03-21 at 8 43 09 AM" src="https://user-images.githubusercontent.com/75961057/159297717-067e84ca-0b50-4593-bb2d-9f2f52236ccf.png">

3. Model evaluation
Result: The target accuracy is 75%. While close to the target, it does not a satisfactorily predict the outcome of the charity donations.
Loss: 0.5550001859664917 and Accuracy: 0.7314285635948181. 
<img width="825" alt="Screen Shot 2022-03-21 at 8 45 18 AM" src="https://user-images.githubusercontent.com/75961057/159299157-56522277-a749-4c26-985f-78e52aca54e1.png">

### Model Optimization
To increase the performance of the model, it is optimized in order to achieve a target predictive accuracy higher than 75%. The following steps are taken:

**Dropping more or fewer columns.**
* The STATUS feature had only 5 non active projects, so it was dropped. 
<img width="804" alt="Screen Shot 2022-03-21 at 8 55 09 AM" src="https://user-images.githubusercontent.com/75961057/159300025-ca79edc4-1115-41fb-ae5e-7894743f99b9.png">
* The SPECIAL CONSIDERATION feature has only 27 organizations out of 34299 that wanted special considerations.
<img width="809" alt="Screen Shot 2022-03-21 at 8 57 59 AM" src="https://user-images.githubusercontent.com/75961057/159300555-947462ad-2e4e-4042-97f9-c58f2167ea8c.png">

**Adding more neurons to a hidden layer.**
* The first hidden layer has 160 neurons, the second layer has 60. 
* Activation function ReLU for the hidden layers and activation function for the output layer is Sigmoid reamained the same. 
* The optimizer adam and loss function binary_crossentropy. remained the same too. 
* Epochs:50

Results: The accuracy dropped. 
Loss: 0.5564196705818176, Accuracy: 0.7286297082901001
<img width="805" alt="Screen Shot 2022-03-21 at 9 08 45 AM" src="https://user-images.githubusercontent.com/75961057/159302787-3fc007fd-c177-4eb3-b26e-086852ece59d.png">

**Changing the activation function¶**
* Hidden layers:  first layer has 80 neurons, the second has 30. 
* Activation function changed to tanh for both the hidden layers. The activation function for the output layer is Sigmoid.
* For the compilation, the optimizer is adam and the loss function is binary_crossentropy.
* Epochs:100

Results: The accuracy dropped.
Loss: 0.5599421262741089, Accuracy: 0.728396475315094
<img width="805" alt="Screen Shot 2022-03-21 at 9 23 19 AM" src="https://user-images.githubusercontent.com/75961057/159305385-0617c795-e900-43d2-89b3-0c58e1073079.png">

**Adding another hidden layer**
* The first hidden layer has 160 neurons, the second layer has 100 neurons, third layer has 60 neurons.
* Activation function ReLU for the hidden layers and activation function for the output layer is Sigmoid reamained the same. 
* The optimizer adam and loss function binary_crossentropy. remained the same too. 
* Epochs:50

Results: The accuracy improved to 73% from the last two optimization attempts but is the same as the initial model. 
Loss: 0.5613805055618286, Accuracy: 0.7301457524299622
<img width="803" alt="Screen Shot 2022-03-21 at 9 29 10 AM" src="https://user-images.githubusercontent.com/75961057/159307731-81c161b5-3591-4490-a21b-1e3c182b7552.png">

# Summary
The deep learning neural network model got a 73% accuracy and did not reach the target of 75%. A Random Forest Classifier could work better than neural networks. This is because random forest is a robust model due to their sufficient number of estimators and tree depth. Also the random forest models have a faster performance than neural networks and could have avoided the data from being overfitted.
