import numpy as np 
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.preprocessing import PolynomialFeatures

############ OLS model ################
def ols_model(y,x, data, baseline):
    # Select target and feature
    model = ols('y ~ x', data).fit()
    # Calculate residuals/error
    evaluate = pd.DataFrame()

    # setting up x independent variable
    evaluate['x'] = x

    # setting up y dependent variable
    evaluate['y'] = y

    # setting up the baseline. baseline was established above
    evaluate['baseline'] = baseline

    # setting up y-hat AKA the predicted y-values. Just introduce model
    evaluate['yhat'] = model.predict()

    # calculate the baseline residuals (residual - actual y)
    evaluate['baseline_residuals'] = evaluate.baseline - evaluate.y

    # calculate the model residuals (y-hat - model)
    evaluate['model_residual'] = evaluate.yhat - evaluate.y
    
    # Baseline sum of squared errors
    baseline_sse = (evaluate.baseline_residuals ** 2).sum()

    # Model sum of squared errors
    model_sse = (evaluate.model_residual ** 2).sum()

    # Checking if our model beats our baseline
    if model_sse < baseline_sse:
        print('Our model out performed our baseline')
        print('It makes sense to evaluate this model more thoroughly')
    else:
        print('The baseline out performed the model')
    
    print('Baseline SSE', baseline_sse)
    print('Model SSE', model_sse)
    return y, evaluate

    ############### Plotting residuals ###############

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    # ls line style allows dotted line
    plt.hlines(0, actual.min(), actual.max(), ls=':')   
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

  ###################### Linear Regression ####################
def linear_regression(X_train_scaled, y_train):
    # Fit the model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_scaled, y_train)
    # predicting out training observations
    lm_pred = lm.predict(X_train_scaled)
    # compute root mean squared error
    lm_mse = median_absolute_error(y_train, lm_pred)
    print(f'The models median absolute error is {round(lm_mse,3)}')
    return lm_pred

    ###################### Laso Lars ####################
def laso_lars(X_train_scaled, y_train):
    # Fit the model
    lars = LassoLars(alpha=1)
    lars.fit(X_train_scaled, y_train)
    # predicting out training observations
    lars_pred = lars.predict(X_train_scaled)
    # compute root mean squared error
    lars_mse = median_absolute_error(y_train, lars_pred)**(1/2)
    print(f'The models median absolute error is {round(lars_mse,3)}')

    ###################### Polynomial Features ####################
def polynomial_features(X_train_scaled, y_train):
    # make the polynomial thing
    pf = PolynomialFeatures(degree=2)
    # fit and transform the thing to get a new set of features..which are the original features sqauared
    X_train_squared = pf.fit_transform(X_train_scaled)
    # Fit the model
    lm_squared = LinearRegression()
    lm_squared.fit(X_train_squared, y_train)
    # predicting out training observations
    lm_squared_pred = lm_squared.predict(X_train_squared)
    # compute root mean squared error. Evaluate
    lm_squared_mse = median_absolute_error(y_train, lm_squared_pred)
    print(f'The models median absolute error is {round(lm_squared_mse,3)}')