# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```python
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks
scored by students.
2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents
the marks scored.
3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents
 where the line crosses the y-axis.
4.Use the linear equation to predict marks based on the input Predicted Marks = m*(hours studied) + b.
5.For each data point calculate the difference between the actual and predicted marks.
6.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update
 these parameters based on the calculated error.
7.Once the model parameters are optimized, use the final equation to predict marks for any new input data.
```

## Program:
```PYTHON
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:   SHYAM SUJIN U
RegisterNumber: 212223040201
```


```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

##plotting for training data
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

##plotting for test data
plt.scatter(x_test,y_test,color="grey")
plt.plot(x_test,y_pred,color="purple")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

### Head :
![image](https://github.com/user-attachments/assets/7415d1c5-e4b8-47c7-811a-217ada324d12)
### Tail :
![image](https://github.com/user-attachments/assets/31cf813b-275a-4d65-99b8-75a4ec620916)
### X Values :
![image](https://github.com/user-attachments/assets/ccea127f-a128-46ad-8c94-36fe03d255aa)
### Y Values :
![image](https://github.com/user-attachments/assets/03db82b0-d1ba-4697-9848-5831e579a35b)
### Y_Prediction Values :
![image](https://github.com/user-attachments/assets/119ec830-e70a-427f-a55b-cba9517dca97)
### Y_Test Values :
![image](https://github.com/user-attachments/assets/7e6558cb-2434-4ae1-825e-a026e2a76aa5)
### MSE,MAE AND RMSE:
![image](https://github.com/user-attachments/assets/3c4a3bd5-ce86-4a36-a9b4-58a0e483d42d)
### Training Set:
![image](https://github.com/user-attachments/assets/159e043f-6e36-407f-857c-3a2b147e4bbc)
### Testing Set:
![image](https://github.com/user-attachments/assets/194759b3-e4f0-423e-8286-fbf5b5479391)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
