import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('HousingData.csv')

#Multiple Linear Regrassion
x = pd.DataFrame(data.iloc[:, :-1])  # features
y = pd.DataFrame(data.iloc[:, -1])   # target

# Remove rows where either X or Y has NaN
x = x.reset_index(drop=True)
y = y.reset_index(drop=True)

# Combine and drop nulls together
combined = pd.concat([x, y], axis=1).dropna()

# Separate again
x = combined.iloc[:, :-1]
y = combined.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"X Train shape is :{X_train.shape}")
print(f"Y Train shape is :{Y_train.shape}")
print(f"X Train shape is :{X_train.shape}")
print(f"Y Train shape is :{Y_train.shape}")

print('_'*30)
print(f"Xtestshape is :{X_test.shape}")
print(f"Y test shape is :{Y_test.shape}")

#Create the model
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#Fit the model
model.fit(X_train, Y_train)
print(model.coef_)#value of m
print(model.intercept_)#value of c

#predictions
y_pred=model.predict(X_test)
y_pred=pd.DataFrame(y_pred,columns=['Perdicted'])
print(y_pred)

#solve the errors 
from sklearn import metrics
print(f"MAE:{metrics.mean_absolute_error(Y_test,y_pred)}")
print(f"MSE:{metrics.mean_squared_error(Y_test,y_pred)}")
print(f"RMSE:{np.sqrt(metrics.mean_squared_error(Y_test,y_pred))}")
print(f"RSqured:{metrics.r2_score(Y_test,y_pred)}")

plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)  # Perfect line
plt.show()