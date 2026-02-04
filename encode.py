
import pandas as pd 

#Data preparation 

data = { 
    'ID' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Size' : [34, 23, 45, 56, 67, 78, 89, 90, 21, 32],
    'Energy Rating' : ['A' , 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Price' : [200000, 150000, 250000, 300000, 220000, 270000, 320000, 210000, 180000, 230000]
}

df = pd.DataFrame(data)
df = df.drop(columns=['ID']) # Drop ID column as it's not needed for analysis
df

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Feature and target without energy rating
x = df.drop(columns=['Price', 'Energy Rating'])
y = df['Price']

# %%
#Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Feature scaling
scaler = StandardScaler()
x_trained_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Model training
model_no_energy = LinearRegression()
model_no_energy.fit(x_trained_scaled, y_train)

# %%
#Prediction and evaluation
y_pred_no_energy = model_no_energy.predict(x_test_scaled)

# Evaluate the model
mse_no_energy = mean_squared_error(y_test, y_pred_no_energy)
print(f'Mean Squared Error without Energy Rating: {mse_no_energy:.2f}')

r2_no_energy = model_no_energy.score(x_test_scaled, y_test)
print(f'R^2 Score without Energy Rating: {r2_no_energy:.2f}')

# %%
#Label Encoding for 'Energy Rating'

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Energy Rating Encoded'] = label_encoder.fit_transform(df['Energy Rating'])
df

# %%
#Feature Engineering with encoded 'Energy Rating'
x_encoded = df.drop(columns=['Price', 'Energy Rating'])
y_encoded = df['Price']

#Train-test split
x_train_enc, x_test_enc, y_train_enc, y_test_enc = train_test_split(
    x_encoded, y_encoded, test_size=0.2, random_state=42)

#Feature scaling
x_train_enc_scaled = scaler.fit_transform(x_train_enc)
x_test_enc_scaled = scaler.transform(x_test_enc)

#Model training with encoded feature
model_with_energy = LinearRegression()
model_with_energy.fit(x_train_enc_scaled, y_train_enc)

#Prediction and evaluation with encoded feature
y_pred = model_with_energy.predict(x_test_enc_scaled)

# Evaluate the model
final_mse = mean_squared_error(y_test_enc, y_pred) 
print(f'Mean Squared Error with Energy Rating: {final_mse:.2f}')

final_r2 = model_with_energy.score(x_test_enc_scaled, y_test_enc)
print(f'R^2 Score with Energy Rating: {final_r2:.2f}')


# %%
#One Hot Encoding for 'Energy Rating'

#Just apply onehot encoding to the original dataframe
df_onehot = pd.get_dummies(df, columns=['Energy Rating'], prefix='Energy')

#Ensuring 
df_onehot = df_onehot.astype(int) # Convert all columns to integer type 

#Display the modified dataframe
df_onehot


