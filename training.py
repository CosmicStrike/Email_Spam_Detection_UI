import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def model_fit_and_test(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
    # n_estimator - number of trees (decesion tree) in forest
    model = RandomForestClassifier(n_estimators=100)

    model.fit(x_train, y_train)

    y_preds = model.predict(x_test.values)
    rf_accuracy = accuracy_score(y_test, y_preds)

    return model, rf_accuracy




if __name__ == '__main__':
    import dataCleaning, pandas as pd
    df = pd.read_csv("spam.csv")
    dataCleaning.clean(df)