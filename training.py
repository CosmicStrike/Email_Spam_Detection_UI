import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


def TF_IDF(df: pd.DataFrame):  # Takes the dataframe with columns= Length, Processed_Text
    TF = TfidfVectorizer(use_idf=True, max_features=3000)
    feq = pd.DataFrame(TF.fit_transform(df["Processed_Text"]).todense(), columns=[
                       TF.get_feature_names_out()])
    print(TF.get_feature_names_out())
    feq["Length"] = np.array(df["Length"])
    feq["Length"].replace('NaN', 230, inplace=True)
    # print("Count: ",feq["Length"].isna().sum())
    # print(feq)
    return feq


def model_fit_and_test(df: pd.DataFrame):
    y = df["labels"]
    x = df.drop(columns="labels", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=10)

    x_train = TF_IDF(x_train)
    x_test = TF_IDF(x_test)

    # n_estimator - number of trees (decesion tree) in forest
    model = RandomForestClassifier(n_estimators=100)

    model.fit(x_train, y_train)

    y_preds = model.predict(x_test.values)
    rf_accuracy = accuracy_score(y_test, y_preds)

    print(y_preds, rf_accuracy)

    return model, rf_accuracy


def test():
    import dataCleaning
    import pandas as pd
    df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
    dataCleaning.initialize()
    dataCleaning.clean(df)

    df = pd.read_csv("cleaned.csv")
    model_fit_and_test(df)


if __name__ == '__main__':
    test()
