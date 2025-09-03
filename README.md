# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: CHANDRU.K
RegisterNumber: 212224220017
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
DATA INFORMATION

<img width="612" height="210" alt="image" src="https://github.com/user-attachments/assets/30b3bd9c-f4a0-4cdb-a431-5039f565ce21" />

Value of X

<img width="272" height="698" alt="Screenshot 2025-08-30 133535" src="https://github.com/user-attachments/assets/8f00d49a-14bc-4b51-ab7b-e4abb1d793a1" />

Value of X1 Sacle

<img width="347" height="701" alt="Screenshot 2025-08-30 133553" src="https://github.com/user-attachments/assets/fac9a01b-99ea-4004-a3b4-e64bbe85bd51" />


predicted values:

<img width="322" height="45" alt="Screenshot 2025-08-30 133603" src="https://github.com/user-attachments/assets/5f5ba842-d597-4a58-ace2-50318e5c9445" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
