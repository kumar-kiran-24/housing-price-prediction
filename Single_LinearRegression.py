import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('HousingData.csv')
# print(data.head())
# print(data.describe())
# print(data.shape)

#Liner Regresssion with 1 varible(ind=lstat,dep=medv)
df_1feat=data[['LSTAT','MEDV']]
df_1feat = df_1feat.dropna()#drop the null valus in data


df_1feat.plot(x='LSTAT',y='MEDV',style='o')
plt.xlabel('Last Value')
plt.ylabel('Medium value of')
# plt.show()

#segrete the depdent and idepdent varibles 
x=pd.DataFrame(df_1feat[['LSTAT']])
y=pd.DataFrame(df_1feat[['MEDV']])

#Divide the data set into train and set
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
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

#Predictions
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

# User Input Prediction
print("\n--- Enter details to predict house price ---")
user_data = []
feature_names = x.columns.tolist()


for col in feature_names:
    value = float(input(f"Enter value for {col}: "))
    user_data.append(value)

# Convert to DataFrame for prediction
user_df = pd.DataFrame([user_data], columns=feature_names)

predicted_price = model.predict(user_df)[0]
print(f"\nPredicted House Price: {predicted_price:.2f}")