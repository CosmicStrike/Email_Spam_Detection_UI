import pandas as pd
import re
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

re_pattren = r'\b[A-Za-z0-9]+\b'
def RE(s:str) -> str:
    a = re.findall(re_pattren,s)
    ans = ' '.join(a)
    return ans

nltk.download("wordnet") #--- lemmatization
nltk.download("punkt") #--- tokenizor 
nltk.download("stopword") 
nltk.download('omw-1.4') #---lemmatization


print()
print()

sampleRow = 6

df = pd.DataFrame(pd.read_csv("spam.csv",encoding="ISO-8859-1"))
print(len(df))
# Check for empty columns
def Remove_Empty_Columns(df):
    empty_cols = []
    for col in df.columns:
        if df[col].isnull().sum():
            empty_cols.append(col)

    # drop empty columns
    df.drop(columns=empty_cols, axis=1, inplace=True)

Remove_Empty_Columns(df)
print("Original Text -> ", df["v2"][sampleRow])



# Length of message is also important factor
def Calculate_Length(df):
    leng = []
    # print("COunt : ", df["v2"].count())
    for i in range(df["v2"].count()):
        leng.append(len(df["v2"][i]))
    df["Length"] = leng

Calculate_Length(df)
print("Length of Text -> ", df["Length"][sampleRow])



# Get the data with only alphanumeric characters; Remove quotes, commas,....
def Convert_To_AlphaNumeric(df):
    l = []
    for i in range(df["v2"].count()):
        l.append(RE(df["v2"][i]).lower())# Convert the data into lower case
    df["Message"] = l

Convert_To_AlphaNumeric(df)
print("Only AlphaNumeric Text -> ", df["Message"][sampleRow])



from nltk import word_tokenize
def Tokenize(df):
    tokens = []
    for i in range(df["Message"].count()):
        tokens.append(word_tokenize(df["Message"][i]))
    df["tokens"] = tokens
Tokenize(df)
print("Tokenized Text -> ", df["tokens"][sampleRow])


from nltk.corpus import stopwords
def RemoveStopword(df):
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

RemoveStopword(df)
print("Stopwords Removed -> ", df["Filtered-Stopwords"][sampleRow])


lem = nltk.WordNetLemmatizer()
def Lemmantize_Text(df):
    lemmantized = []
    l = []
    for i in range(df["Filtered-Stopwords"].count()):
        l = []
        for word in df["Filtered-Stopwords"][i]:
            l.append(lem.lemmatize(word))
        lemmantized.append(l)

    df["Lemmanted"] = lemmantized

Lemmantize_Text(df)

print("Lemmatization -> ", df["Lemmanted"][sampleRow])


def FinalizeText(df):
    text = []
    for i in range(df["Lemmanted"].count()):
        t = ' '.join(df["Lemmanted"][i])
        text.append(t)
    df["Processed_Text"] = text

FinalizeText(df)
print("Original text -> ",df["v2"][sampleRow])
print("Processed text -> ",df["Processed_Text"][sampleRow])


#Encoding the output using label encoder
label_encoder = LabelEncoder()
encoded = label_encoder.fit_transform(df["v1"])
df["labels"] = encoded


def RemoveUnwantedColumn(df):
    df.drop(['v1','v2', 'Message','tokens', 'Filtered-Stopwords','Lemmanted'], axis=1, inplace=True)
RemoveUnwantedColumn(df)

print(df.columns)


y = df["labels"]
x = df.drop(columns="labels", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=10)

def TF_IDF(dataframe):# Takes the dataframe with columns= Length, Processed_Text
    TF = TfidfVectorizer(use_idf=True, max_features=3000)
    feq = pd.DataFrame(TF.fit_transform(dataframe["Processed_Text"]).todense(), columns=[TF.get_feature_names_out()])
    print(TF.get_feature_names_out())
    feq["Length"] = np.array(dataframe["Length"])
    feq["Length"].replace('NaN',230, inplace=True)
    # print("Count: ",feq["Length"].isna().sum())
    # print(feq)
    return feq

xtrain = TF_IDF(x_train)
xtest = TF_IDF(x_test)

print(xtrain.shape)
print(xtest.shape)




# def PreProcess_Data(dataf):
#     Calculate_Length(dataf)
#     Tokenize(dataf)
#     RemoveStopword(dataf)
#     Lemmantize_Text(dataf)
#     FinalizeText(dataf)
#     RemoveUnwantedColumn(dataf)
#     return Vectorize_And_TFIDFTranform(dataf)
    