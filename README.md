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

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,y_pred,color="yellow")

plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
<img width="424" alt="image" src="https://github.com/shrenidhi28/ML2---Simple-Linear-Regression/assets/155261096/5c19eb05-b9b4-4150-a76a-97970dd1d97c">



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
