from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import re
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

re_pattren = r'\b[A-Za-z0-9]+\b'

lem = nltk.WordNetLemmatizer()
label_encoder = LabelEncoder()


def RE(s: str) -> str:
    a = re.findall(re_pattren, s)
    ans = ' '.join(a)
    return ans


def initialize():
    # --- lemmatization
    nltk.download("wordnet", download_dir="./nltk_downloads")
    nltk.download("punkt", download_dir="./nltk_downloads")  # --- tokenizor
    nltk.download("stopword", download_dir="./nltk_downloads")
    # ---lemmatization
    nltk.download('omw-1.4', download_dir="./nltk_downloads")

    nltk.data.path.append('./nltk_downloads')


sampleRow = 6


def load_dataset() -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv("spam.csv", encoding="ISO-8859-1"))
    return df


# Check for empty columns
def Remove_Empty_Columns(df: pd.DataFrame):
    empty_cols = []
    for col in df.columns:
        if df[col].isnull().sum():
            empty_cols.append(col)
    # drop empty columns
    df.drop(columns=empty_cols, axis=1, inplace=True)

# Length of message is also important factor


def Calculate_Length(df: pd.DataFrame):
    leng = []
    # print("COunt : ", df["v2"].count())
    for i in range(df["v2"].count()):
        leng.append(len(df["v2"][i]))
    df["Length"] = leng


# Get the data with only alphanumeric characters; Remove quotes, commas,....
def Convert_To_AlphaNumeric(df: pd.DataFrame):
    l = []
    for i in range(df["v2"].count()):
        l.append(RE(df["v2"][i]).lower())  # Convert the data into lower case
    df["Message"] = l


def Tokenize(df: pd.DataFrame):
    tokens = []
    for i in range(df["Message"].count()):
        tokens.append(word_tokenize(df["Message"][i]))
    df["tokens"] = tokens


def RemoveStopword(df: pd.DataFrame):
    stop_words = set(stopwords.words("english"))
    fil = []
    filters = []
    for i in range(df["tokens"].count()):
        fil = []
        for word in df["tokens"][i]:
            if word.casefold() not in stop_words:
                fil.append(word)
        filters.append(fil)
    df["Filtered-Stopwords"] = filters


def Lemmantize_Text(df: pd.DataFrame):
    lemmantized = []
    l = []
    for i in range(df["Filtered-Stopwords"].count()):
        l = []
        for word in df["Filtered-Stopwords"][i]:
            l.append(lem.lemmatize(word))
        lemmantized.append(l)

    df["Lemmanted"] = lemmantized


def FinalizeText(df: pd.DataFrame):
    text = []
    for i in range(df["Lemmanted"].count()):
        t = ' '.join(df["Lemmanted"][i])
        text.append(t)
    df["Processed_Text"] = text


def RemoveUnwantedColumn(df):
    df.drop(['v1', 'v2', 'Message', 'tokens', 'Filtered-Stopwords',
            'Lemmanted'], axis=1, inplace=True)


def TF_IDF(df: pd.DataFrame):  # Takes the dataframe with columns= Length, Processed_Text
    TF = TfidfVectorizer(use_idf=True, max_features=3000)
    feq = pd.DataFrame(TF.fit_transform(
        df["Processed_Text"]).todense(), columns=[TF.get_feature_names_out()])
    print(TF.get_feature_names_out())
    feq["Length"] = np.array(df["Length"])
    feq["Length"].replace('NaN', 230, inplace=True)
    # print("Count: ",feq["Length"].isna().sum())
    # print(feq)
    return feq


def FitData(dataf, size):
    for i in range(dataf.shape[1], size+1):
        dataf[f"{i}"] = 0.0


def clean(df: pd.DataFrame) -> list[np.array]:
    Remove_Empty_Columns(df)
    Calculate_Length(df)
    Convert_To_AlphaNumeric(df)
    Tokenize(df)
    RemoveStopword(df)
    Lemmantize_Text(df)
    FinalizeText(df)
    encoded = label_encoder.fit_transform(df["v1"])
    df["labels"] = encoded
    RemoveUnwantedColumn(df)

    y = df["labels"]
    x = df.drop(columns="labels", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=10)

    xtrain = TF_IDF(x_train)
    xtest = TF_IDF(x_test)

    return xtrain, xtest, y_train, y_test
