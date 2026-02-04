import pandas as pd 
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

#Convert to DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names) 
#Add target variable to DataFrame 
data['MedHouseVal'] = california.target

#MedHouseVal = y 
#rest of the features = X

data.head()

#Data Preprocessing 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#define feature variables and target variable
x = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

#split the data into training and testing sets 
# #important to evaluate the model performance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

#feature scaling
#important for algorithm
#solution to fit the model faster
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Train the Linear Regression Model
from sklearn.linear_model import LinearRegression #Ordinary Least Squares

#create the model
model = LinearRegression()

#Train the model on the training data
model.fit(x_train_scaled, y_train)

#Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

#Make Predictions and Evaluate the Model 
from sklearn.metrics import mean_squared_error, r2_score

#Make predictions on the test data
y_pred = model.predict(x_test_scaled)

#Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error:{mse:.4f}")
print(f"R-squared: {r2:.4f}")
print(y_pred)
print("Not Quite Accurate")

# %%
#Visualize the Results
import matplotlib.pyplot as plt

#Plotting Actual vs Predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color = 'purple', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Actual vs Predicted Median House Values')
plt.show()
