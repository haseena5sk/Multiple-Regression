import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#Loading data set
house_df = pd.read_csv('housing.csv')
print("Housing dataset columns:")
print(house_df.columns)
#Feature Extraction
X = house_df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area
Number of Rooms',
'Avg. Area Number of Bedrooms', 'Area Population']]
y = house_df['Price']
#Splitting Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=
0.3)
# Linear Regresion Model Creation
reg_model = LinearRegression()
#Traing the model (fit)
reg_model.fit(X_train, y_train)
#Model Prediction
y_pred = reg_model.predict(X_test)
#Accuracy of model
print("R-Sqaure value", r2_score(y_test, y_pred))
#Visualization
plt.scatter(y_test, y_pred)
plt.show()