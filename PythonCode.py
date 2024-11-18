import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

#12 items per row
df = pd.read_csv('heart.csv')

#This Defines the predictor and response variables
X = df[['RestingBP', 'Cholesterol', 'MaxHR']]
y = df['HeartDisease']

#This is K-fold cross-validation
cv = KFold(n_splits=10, random_state=1, shuffle=True)

#model = GaussianNB()
model = RandomForestClassifier()

# Evaluate model using mean absolute error (MAE)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
mean_absolute_error = np.mean(np.abs(scores))

# Calculate root mean squared error (RMSE)
scores_rmse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
rmse = np.sqrt(np.mean(np.abs(scores_rmse)))

print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")