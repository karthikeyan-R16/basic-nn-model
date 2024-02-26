# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The problem statement for developing a neural network regression model involves predicting a continuous value output based on a set of input features. In regression tasks, the goal is to learn a mapping from input variables to a continuous target variable.


## Neural Network Model

Include the neural network model diagram.

![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/f6e31be3-072d-46cf-9772-077475503592)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Karthikeyan R
### Register Number: 212222240045
```python
#DEPENDENCIES:

from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

#DATA FROM SHEETS:

worksheet = gc.open("DL ex 1").sheet1
rows=worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
print(df)

df.head()

#DATA VISUALIZATION:

 x = df[["Input"]] .values
 y = df[["Output"]].values

#DATA SPLIT AND PREPROCESSING:

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

print(x_train)
print(x_test)

#REGRESSIVE MODEL:

 model = Seq([
 Den(4,activation = 'relu',input_shape=[1]),
 Den(6),
 Den(3,activation = 'relu'),
 Den(1),
 ])

 model.compile(optimizer = 'rmsprop',loss = 'mse')
 model.fit(x_train,y_train,epochs=20)
 model.fit(x_train,y_train,epochs=20)

#LOSS CALCULATION:

loss_plot = pd.DataFrame(model.history.history)
loss_plot.plot()

 err = rmse()
 preds = model.predict(x_test)
 err(y_test,preds)

 x_n1 = [[30]]
 x_n_n = scaler.transform(x_n1)
 model.predict(x_n_n)

#PREDICTION:

y_pred=model.predict(x_test)
y_pred

```
## Dataset Information


![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/07436956-fc8b-4c23-ba10-fa82fb22967f)



## OUTPUT



![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/b96fa92c-bb54-4a6b-a335-d381650bc089)



### value of X_train and X_test:


![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/801db674-e879-40c3-b655-cae23632f542)



### ARCHITECTURE AND TRAINING:


![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/d486b700-fea0-44d6-a0df-ba17c82e2adc)




![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/9a772a75-926d-40ed-8549-b6cdde9c332c)



### Training Loss Vs Iteration Plot


![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/c0aa6d53-2148-4de7-9951-48401ee9e83d)



### Test Data Root Mean Squared Error
![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/fa2f1986-96cb-4c3b-a140-382e8b6b4eab)


![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/48c9be6c-57d6-4e7d-b81a-e7b6c559e542)



### New Sample Data Prediction

![image](https://github.com/karthikeyan-R16/basic-nn-model/assets/119421232/d579ae98-93fc-4a09-99b3-1eb7d018f424)


## RESULT
A neural network regression model for the given dataset is developed .


