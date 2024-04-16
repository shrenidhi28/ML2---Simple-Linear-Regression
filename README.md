# ML2---Simple-Linear-Regression

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook


## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv("./student_scores.csv")
df.head()

x_hours = df.iloc[:,:-1].values
y_scores = df.iloc[:,1].values

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
y_pred


GRAPH FOR TRAINING DATA

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,y_pred,color="yellow")

plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


ERROR

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)


GRPAH FOR TESTING DATA

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
GRAPH FOR TRAINING DATA <br>

<img width="424" alt="image" src="https://github.com/shrenidhi28/ML2---Simple-Linear-Regression/assets/155261096/5c19eb05-b9b4-4150-a76a-97970dd1d97c">
<br>
<br>

GRPAH FOR TESTING DATA <br>

<img width="424" src="https://github.com/shrenidhi28/ML2---Simple-Linear-Regression/assets/155261096/aa777a81-373d-4a5c-968e-fa11fff632cd" alt="image">
<br>
<br>

ERROR

<img src="https://github.com/shrenidhi28/ML2---Simple-Linear-Regression/assets/155261096/1d476efa-5915-47d9-98c3-a9d7a4014c95" width="424" alt="image">







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
