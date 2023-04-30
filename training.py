import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import dataCleaning
import pandas as pd


# Takes the dataframe with columns= Length, Processed_Text
def TF_IDF_train(TF, df: pd.DataFrame):
    df.fillna("a", inplace=True)
    feq = pd.DataFrame(TF.fit_transform(df["Processed_Text"]).todense(), columns=[
                       TF.get_feature_names_out()])
    print(TF.get_feature_names_out())
    feq["Length"] = np.array(df["Length"])
    feq["Length"].replace('NaN', 230, inplace=True)
    # print("Count: ",feq["Length"].isna().sum())
    # print(feq)
    save_vectorizer(TF)
    return feq


# Takes the dataframe with columns= Length, Processed_Text
def TF_IDF_test(TF, df: pd.DataFrame):
    df.fillna("a", inplace=True)
    feq = pd.DataFrame(TF.transform(df["Processed_Text"]).todense(), columns=[
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

    TF = TfidfVectorizer(use_idf=True, max_features=3000)
    x_train = TF_IDF_train(TF, x_train)
    x_test = TF_IDF_test(TF, x_test)

    # n_estimator - number of trees (decesion tree) in forest
    model = RandomForestClassifier(n_estimators=100)

    model.fit(x_train, y_train)

    y_preds = model.predict(x_test.values)
    rf_accuracy = accuracy_score(y_test, y_preds)
    save_model(model)
    print(rf_accuracy)
    return rf_accuracy


def save_model(model):
    filename = "model.pickle"
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model():
    filename = "model.pickle"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return False


def save_vectorizer(TF):
    pickle.dump(TF, open("vectorizer.pickle", "wb"))


def load_vectorizer():
    return pickle.load(open("vectorizer.pickle", 'rb'))


def test():
    df = pd.read_csv("spam.csv", encoding="ISO-8859-1")
    dataCleaning.initialize()
    dataCleaning.clean(df)

    df = pd.read_csv("cleaned.csv")
    model_fit_and_test(df)


def predict(email):
    df = pd.DataFrame({
        "v2": [email],
        "v1": [0]
    })
    dataCleaning.initialize()
    dataCleaning.Calculate_Length(df)
    dataCleaning.Convert_To_AlphaNumeric(df)
    dataCleaning.Tokenize(df)
    dataCleaning.RemoveStopword(df)
    dataCleaning.Lemmantize_Text(df)
    dataCleaning.FinalizeText(df)
    dataCleaning.RemoveUnwantedColumn(df)

    TF = load_vectorizer()

    df = TF_IDF_test(TF, df)

    model: RandomForestClassifier = load_model()
    if model:
        return [model.predict(df),model.predict_proba(df)]

    return 0


if __name__ == '__main__':
    pass
