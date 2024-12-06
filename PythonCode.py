import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LinearRegression

#12 items per row
df = pd.read_csv('heart.csv')

Z = df.apply(LabelEncoder().fit_transform)
#5 categorical items for encoder
check=True # Used for testing the code
#This Defines the predictor and response variables
if check==True:
    X = Z[['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak','Age','FastingBS',
           'Age','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']]
    y = Z['HeartDisease']

    #This is K-fold cross-validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    #Change which algorithm by removing/adding a # at begining of 'model'
    model = GaussianNB()
    #model = RandomForestClassifier()

    # Evaluate model using mean absolute error (MAE)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    mean_absolute_error = np.mean(np.abs(scores))

    # Calculate root mean squared error (RMSE)
    scores_rmse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse = np.sqrt(np.mean(np.abs(scores_rmse)))

    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Get predictions and show results
    predictions = []
    actuals = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #print(confusion_matrix(y_pred, y_test))
        
        predictions.extend(y_pred)
        actuals.extend(y_test)

        print(f"Fold accuracy: {accuracy_score(y_test, y_pred)}")

    overall_accuracy = accuracy_score(actuals, predictions)
    print(f"Overall accuracy: {overall_accuracy}")
    # Calculate Confusion Matrix
    conMtrx = confusion_matrix(predictions, actuals)
    tn, fp, fn, tp = conMtrx.ravel()
    print(conMtrx)
    print((tp, fp, tn, fn))