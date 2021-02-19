import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def split_data(df):
    x = df[["area", "flat", "house", "new_dev", "penthouse"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

    return  X_train, X_test, y_train, y_test

def linear_regression(X_train, X_test, y_train, y_test):
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    return r2, y_test, y_pred
